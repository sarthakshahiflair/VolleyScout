# -*- coding: utf-8 -*-
"""
Trajectory-based ball auto-labelling (independent of the COCO "sports ball" teacher).

Why: the current ball model is trained on COCO pseudo-labels, so it can never
learn balls the teacher misses.  This module produces labels from an
INDEPENDENT signal -- motion + physics:

  1. candidates   background subtraction (static camera) -> small moving blobs,
                  colour score, optional low-confidence detector boxes
  2. tracking     Hungarian association with constant-velocity prediction,
                  gap tolerance
  3. physics      keep only tracks that contain ballistic arcs
                  (x linear in time, y quadratic, small residual, no upward
                  acceleration).  Static lookalikes, hands, shoes, crowd fail.
  4. interpolate  fill short gaps inside kept tracks
  5. export       YOLO dataset (time-block train/val split) + CVAT-importable
                  YOLO 1.1 folder + review manifest ranked by uncertainty

Auto-labels are a FIRST PASS.  Review the manifest's high-priority rows in CVAT
and keep a hand-labelled test set that is never trained on.

Assumes a STATIC camera (as in the Villadoro videos).
"""
from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

Detector = Callable[[np.ndarray], Sequence[Tuple[float, float, float, float, float]]]


@dataclass
class AutoLabelConfig:
    # candidate generation
    bg_history: int = 250
    bg_var_threshold: float = 28.0
    min_area: int = 10
    max_area: int = 800
    max_aspect: float = 3.8
    roi: Tuple[float, float, float, float] = (0.02, 0.03, 0.98, 0.98)  # x0,y0,x1,y1
    ignore_rects: Tuple[Tuple[float, float, float, float], ...] = ()    # normalised
    max_candidates: int = 60
    ball_hue: Tuple[int, int] = (5, 40)       # yellow / orange balls (OpenCV 0-179)
    ball_min_sat: int = 80
    # tracking
    init_gate_px: float = 90.0               # 1st link of a new track (no velocity yet)
    gate_base_px: float = 35.0
    gate_vel_gain: float = 1.0
    max_gap_frames: int = 6
    # track validation
    min_track_len: int = 8
    min_total_disp_px: float = 80.0
    win: int = 8
    rms_thresh_px: float = 3.5
    min_ballistic_frac: float = 0.60
    min_accel_y: float = -0.15               # px/frame^2 (y grows downward)
    min_window_speed_px: float = 6.0         # px/frame inside a ballistic window (rejects walking players, labels)
    min_peak_speed_px: float = 14.0          # px/frame the track must reach at least once
    interp_max_gap: int = 6
    # box size
    min_box_px: int = 10
    max_box_px: int = 40


# ───────────────────────────── candidates ──────────────────────────────────
def _roi_mask(h: int, w: int, cfg: AutoLabelConfig) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = cfg.roi
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    for (a, b, c, d) in cfg.ignore_rects:
        m[int(b * h):int(d * h), int(a * w):int(c * w)] = 0
    return m


def find_candidates(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    roi: np.ndarray,
    cfg: AutoLabelConfig,
    detector: Optional[Detector] = None,
) -> List[Tuple[float, float, float, float, float]]:
    """Return [(cx, cy, w, h, score)] for one frame."""
    m = cv2.bitwise_and(fg_mask, roi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out: List[Tuple[float, float, float, float, float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > cfg.max_area:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > cfg.max_aspect:
            continue
        patch = hsv[y:y + h, x:x + w]
        hue_ok = (patch[..., 0] >= cfg.ball_hue[0]) & (patch[..., 0] <= cfg.ball_hue[1])
        col = float(np.mean(hue_ok & (patch[..., 1] >= cfg.ball_min_sat)))
        compact = float(area) / max(1.0, w * h)
        score = 0.30 + 0.40 * col + 0.30 * compact
        out.append((float(cents[i][0]), float(cents[i][1]), float(w), float(h), score))
    out.sort(key=lambda c: -c[4])
    out = out[: cfg.max_candidates]
    if detector is not None:
        for (cx, cy, w, h, conf) in detector(frame):
            out.append((float(cx), float(cy), float(w), float(h), 0.5 + 0.5 * float(conf)))
    return out


# ───────────────────────────── tracking ────────────────────────────────────
@dataclass
class Track:
    tid: int
    obs: List[Tuple[int, float, float, float, float, float]] = field(default_factory=list)  # f,x,y,w,h,score
    missed: int = 0

    def velocity(self) -> Tuple[float, float]:
        if len(self.obs) < 2:
            return 0.0, 0.0
        (f0, x0, y0, *_), (f1, x1, y1, *_) = self.obs[-2], self.obs[-1]
        d = max(1, f1 - f0)
        return (x1 - x0) / d, (y1 - y0) / d

    def predict(self, frame: int) -> Tuple[float, float]:
        f, x, y, *_ = self.obs[-1]
        vx, vy = self.velocity()
        d = frame - f
        return x + vx * d, y + vy * d


class MultiTracker:
    def __init__(self, cfg: AutoLabelConfig):
        self.cfg = cfg
        self.active: List[Track] = []
        self.done: List[Track] = []
        self._next = 1

    def step(self, frame: int, cands: Sequence[Tuple[float, float, float, float, float]]) -> None:
        cfg = self.cfg
        used: set = set()
        matched_rows: set = set()
        if self.active and cands:
            cost = np.full((len(self.active), len(cands)), 1e6)
            for i, tr in enumerate(self.active):
                px, py = tr.predict(frame)
                vx, vy = tr.velocity()
                if len(tr.obs) == 1:
                    gate = cfg.init_gate_px * (tr.missed + 1)
                else:
                    gate = cfg.gate_base_px + cfg.gate_vel_gain * math.hypot(vx, vy) * (tr.missed + 1)
                for j, c in enumerate(cands):
                    d = math.hypot(c[0] - px, c[1] - py)
                    if d <= gate:
                        cost[i, j] = d - 6.0 * c[4]
            ri, cj = linear_sum_assignment(cost)
            for i, j in zip(ri, cj):
                if cost[i, j] < 1e5:
                    c = cands[j]
                    self.active[i].obs.append((frame, c[0], c[1], c[2], c[3], c[4]))
                    self.active[i].missed = 0
                    used.add(j)
                    matched_rows.add(i)
        for i, tr in enumerate(self.active):
            if i not in matched_rows:
                tr.missed += 1
        still = []
        for tr in self.active:
            (self.done if tr.missed > cfg.max_gap_frames else still).append(tr)
        self.active = still
        for j, c in enumerate(cands):
            if j not in used:
                t = Track(self._next)
                self._next += 1
                t.obs.append((frame, c[0], c[1], c[2], c[3], c[4]))
                self.active.append(t)

    def finish(self) -> List[Track]:
        self.done.extend(self.active)
        self.active = []
        return self.done


# ───────────────────────────── physics filter ──────────────────────────────
def ballistic_coverage(track: Track, cfg: AutoLabelConfig) -> Tuple[float, float, np.ndarray, float]:
    """(covered fraction, total displacement, covered-mask, peak speed px/frame)."""
    f = np.array([o[0] for o in track.obs], float)
    x = np.array([o[1] for o in track.obs], float)
    y = np.array([o[2] for o in track.obs], float)
    n = len(f)
    covered = np.zeros(n, bool)
    if n < cfg.win:
        return 0.0, 0.0, covered, 0.0
    disp = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    for s0 in range(0, n - cfg.win + 1):
        sl = slice(s0, s0 + cfg.win)
        tt = f[sl] - f[sl][0]
        if tt[-1] < 2:
            continue
        px = np.polyfit(tt, x[sl], 1)
        py = np.polyfit(tt, y[sl], 2)
        rx = x[sl] - np.polyval(px, tt)
        ry = y[sl] - np.polyval(py, tt)
        rms = float(np.sqrt(np.mean(rx ** 2 + ry ** 2)))
        speed = math.hypot(x[sl][-1] - x[sl][0], y[sl][-1] - y[sl][0]) / tt[-1]
        if rms <= cfg.rms_thresh_px and py[0] >= cfg.min_accel_y and speed >= cfg.min_window_speed_px:
            covered[sl] = True
    peak = 0.0
    idx = np.where(covered)[0]
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if b_ == a_ + 1:
            peak = max(peak, math.hypot(x[b_] - x[a_], y[b_] - y[a_]) / max(1.0, f[b_] - f[a_]))
    return float(covered.mean()), disp, covered, peak


def validate_tracks(tracks: Sequence[Track], cfg: AutoLabelConfig) -> List[Tuple[Track, float]]:
    """Keep ballistic, fast-enough tracks and TRIM them to the ballistic part
    (start/end of a track often drift onto hands, legs or labels)."""
    kept = []
    for tr in tracks:
        if len(tr.obs) < cfg.min_track_len:
            continue
        frac, disp, covered, peak = ballistic_coverage(tr, cfg)
        if frac < cfg.min_ballistic_frac or disp < cfg.min_total_disp_px or peak < cfg.min_peak_speed_px:
            continue
        tr.obs = [o for o, c in zip(tr.obs, covered) if c]
        if len(tr.obs) < cfg.min_track_len:
            continue
        score = frac * min(1.0, len(tr.obs) / 20.0)
        kept.append((tr, score))
    return kept


# ───────────────────────────── labels ──────────────────────────────────────
@dataclass
class BallLabel:
    frame: int
    x: float
    y: float
    side: float
    track_id: int
    track_score: float
    source: str            # detected | interpolated
    priority: float = 0.0  # higher = review first
    reason: str = ""


def tracks_to_labels(kept: Sequence[Tuple[Track, float]], cfg: AutoLabelConfig) -> Dict[int, BallLabel]:
    labels: Dict[int, BallLabel] = {}
    for tr, score in kept:
        sides = [min(o[3], o[4]) for o in tr.obs]
        side = float(np.clip(np.median(sides) * 1.3, cfg.min_box_px, cfg.max_box_px))
        frames = [o[0] for o in tr.obs]
        xs = [o[1] for o in tr.obs]
        ys = [o[2] for o in tr.obs]
        n = len(frames)
        for k, o in enumerate(tr.obs):
            edge = k < 3 or k >= n - 3
            pr = (1 - score) + (0.3 if edge else 0.0) + (1 - min(1.0, o[5])) * 0.3
            labels[o[0]] = BallLabel(o[0], o[1], o[2], side, tr.tid, score, "detected",
                                     pr, "track_edge" if edge else ("low_score" if score < 0.7 else ""))
        for a in range(n - 1):
            gap = frames[a + 1] - frames[a] - 1
            if 0 < gap <= cfg.interp_max_gap:
                for g in range(1, gap + 1):
                    r = g / (gap + 1)
                    fr = frames[a] + g
                    labels.setdefault(fr, BallLabel(
                        fr, xs[a] + r * (xs[a + 1] - xs[a]), ys[a] + r * (ys[a + 1] - ys[a]),
                        side, tr.tid, score, "interpolated", 1.0 + (1 - score), "interpolated"))
    return labels


# ───────────────────────────── pipeline ────────────────────────────────────
def autolabel_video(
    video: Path,
    cfg: Optional[AutoLabelConfig] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    warmup_sec: float = 8.0,
    detector: Optional[Detector] = None,
    progress_every: int = 1000,
) -> Tuple[Dict[int, BallLabel], float, Tuple[int, int]]:
    """Pass 1: motion candidates -> tracks -> physics filter -> per-frame labels."""
    cfg = cfg or AutoLabelConfig()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_start = max(0, int((start_sec - warmup_sec) * fps))
    f_label0 = int(start_sec * fps)
    f_end = total if end_sec is None else min(total, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=cfg.bg_history, varThreshold=cfg.bg_var_threshold, detectShadows=False)
    roi = _roi_mask(h, w, cfg)
    tracker = MultiTracker(cfg)
    fi = f_start
    while fi < f_end:
        ok, frame = cap.read()
        if not ok:
            break
        fg = bg.apply(frame)
        if fi >= f_label0:
            tracker.step(fi, find_candidates(frame, fg, roi, cfg, detector))
        fi += 1
        if progress_every and fi % progress_every == 0:
            print(f"  autolabel pass1 frame {fi}/{f_end}")
    cap.release()
    kept = validate_tracks(tracker.finish(), cfg)
    return tracks_to_labels(kept, cfg), fps, (w, h)


def export_dataset(
    video: Path,
    labels: Dict[int, BallLabel],
    fps: float,
    size: Tuple[int, int],
    out_dir: Path,
    label_stride: int = 2,
    max_pos: int = 4000,
    neg_ratio: float = 0.0,
    neg_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    val_block_sec: float = 30.0,
    cvat: bool = False,
    seed: int = 0,
) -> Dict[str, int]:
    """Pass 2: save selected frames + YOLO labels + manifest (+ optional CVAT folder).

    Negatives (empty-label frames) are OFF unless neg_ranges is given.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    pos = sorted(labels)
    pos = [f for k, f in enumerate(pos) if k % max(1, label_stride) == 0]
    if len(pos) > max_pos:
        pos = sorted(rng.choice(pos, max_pos, replace=False).tolist())
    near = set()
    for f in labels:
        near.update(range(f - 5, f + 6))
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = (min(labels), max(labels)) if labels else (0, total)
    cand_neg = [f for f in range(lo, min(hi, total)) if f not in near]
    # Recall of the auto-labeller is limited, so a random frame inside a rally may
    # contain an UNLABELLED ball.  Negatives are therefore only drawn from
    # neg_ranges (dead-ball periods, seconds); with no ranges they are disabled.
    if neg_ranges is None:
        cand_neg = []
    else:
        cand_neg = [f for f in cand_neg if any(a <= f / fps <= b for a, b in neg_ranges)]
    n_neg = min(len(cand_neg), int(neg_ratio * len(pos)))
    neg = sorted(rng.choice(cand_neg, n_neg, replace=False).tolist()) if n_neg else []
    wanted = {f: True for f in pos}
    wanted.update({f: False for f in neg})

    stem = video.stem
    out_dir = Path(out_dir)
    for sp in ("train", "val"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {"train": 0, "val": 0, "neg": len(neg), "pos": len(pos)}
    fi, last = 0, max(wanted) if wanted else -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(wanted) if wanted else 0)
    fi = min(wanted) if wanted else 0
    while fi <= last:
        if not cap.grab():
            break
        if fi in wanted:
            ok, frame = cap.retrieve()
            if ok:
                t = fi / fps
                sp = "val" if int(t // val_block_sec) % 5 == 4 else "train"
                name = f"{stem}_f{fi:07d}"
                cv2.imwrite(str(out_dir / "images" / sp / f"{name}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                lab = out_dir / "labels" / sp / f"{name}.txt"
                if wanted[fi]:
                    b = labels[fi]
                    lab.write_text(f"0 {b.x / w:.6f} {b.y / h:.6f} {b.side / w:.6f} {b.side / h:.6f}\n")
                    manifest.append([f"{name}.jpg", sp, fi, round(t, 3), round(b.x, 1), round(b.y, 1),
                                     round(b.side, 1), b.source, b.track_id, round(b.track_score, 3),
                                     round(b.priority, 3), b.reason])
                else:
                    lab.write_text("")
                    manifest.append([f"{name}.jpg", sp, fi, round(t, 3), "", "", "", "negative", "", "", 0.0, ""])
                counts[sp] += 1
        fi += 1
    cap.release()

    manifest.sort(key=lambda r: -float(r[10] or 0))
    with open(out_dir / "review_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "split", "frame", "time_sec", "x", "y", "box_px", "source",
                     "track_id", "track_score", "review_priority", "reason"])
        wr.writerows(manifest)
    (out_dir / "data_ball_auto.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n",
        encoding="utf-8")
    if cvat:
        export_cvat_folder(out_dir)
    return counts


def export_cvat_folder(out_dir: Path) -> None:
    """YOLO 1.1 layout CVAT can import (zip the 'cvat_yolo' folder)."""
    root = Path(out_dir) / "cvat_yolo"
    data = root / "obj_train_data"
    data.mkdir(parents=True, exist_ok=True)
    lines = []
    for sp in ("train", "val"):
        for img in sorted((Path(out_dir) / "images" / sp).glob("*.jpg")):
            lab = Path(out_dir) / "labels" / sp / f"{img.stem}.txt"
            for src, dst in ((img, data / img.name), (lab, data / lab.name)):
                try:
                    if dst.exists():
                        dst.unlink()
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            lines.append(f"data/obj_train_data/{img.name}")
    (root / "obj.names").write_text("ball\n", encoding="utf-8")
    (root / "obj.data").write_text(
        "classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n", encoding="utf-8")
    (root / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_yolo_detector(weights: str, device=0, conf: float = 0.05, imgsz: int = 1280) -> Detector:
    """Optional: feed a (weak) YOLO ball model's LOW-confidence boxes in as extra candidates."""
    from ultralytics import YOLO  # lazy
    model = YOLO(weights)

    def _det(frame: np.ndarray):
        r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        res = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                res.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf[0])))
        return res

    return _det
