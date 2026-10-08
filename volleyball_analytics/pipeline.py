# -*- coding: utf-8 -*-
"""
VolleyScout analytics pipeline - Phases 1-4:
  Ball detection/tracking, rally/event logic, jersey OCR, player tracking.

Fixes applied (v2):
  1. min_ball_speed lowered → Set / Attack / Block / Dig now fire correctly.
  2. _classify_touch thresholds tuned for real pixel-speed values from 25 fps HD video.
  3. Jersey post-join: after the full video scan the final jersey_cache is merged
     back into every closed event row so jersey_number is never empty when known.
  4. Single clean CSV per video with exactly 11 required fields.

Improvements applied (v3) — jersey identity pipeline:
  5. Best-frame selection: OCR only runs on high-quality frames per track
     (large bbox, low motion blur, good brightness) to reduce garbage votes.
  6. Jersey number range filter: numbers outside [1, max_jersey_number] are rejected
     before voting, eliminating OCR misreads from scoreboards/shirt logos.
  7. Adaptive OCR interval: confirmed tracks skip OCR entirely; unconfirmed tracks
     sample more aggressively when frame quality is good.

Improvements applied (v4) — tighter jersey identity + track clustering:
  8. Confirmation threshold raised: requires 3 agreeing votes AND avg confidence ≥ 0.55
     before a track identity is locked (was: 2 votes, no confidence gate).
  9. Anti-spam voting: if the same jersey number is voted in ≥ 3 consecutive frames
     for a track, it counts as only 1 vote (prevents stationary-player vote flooding).
  10. Upper-body-only crop: torso region restricted to top 55 % of player bbox before
      applying OCR region variants (removes leg/shoe area with sponsor logos).
  11. Post-process track clustering: after full video scan, tracks sharing the same
      jersey number that are temporally adjacent are merged so that downstream events
      inherit the correct jersey even when the tracker fragmented the same physical
      player across multiple track IDs.
  12. Tighter OCR quality gates: min bbox area raised to 5 000 px², blur threshold to
      60.0, minimum per-vote confidence to 0.50 — all configurable via PipelineConfig.

Improvements applied (v5) — event logic correctness + jersey isolation:
  13. Jersey state fully reset per video: JerseyReader votes/confirmed dicts and
      jersey_cache are cleared at the start of each process_video() call so that
      track IDs from a previous video can never leak into the next one (was Bug #1).
  14. Reception vs Dig post-processing rule: events immediately following a Serve
      are relabelled Reception (not Dig) as a first-ball contact must be a Reception.
  15. Serve fallback rule: if a Rally event is not followed by a Serve within 5 s,
      the first detected contact in that rally is retroactively relabelled Serve.
  16. Block cleanup: consecutive Blocks without a preceding Attack in ≤ 2 s are
      relabelled Attack (a Block without an Attack is physically impossible).
  17. Rally duration cap: a rally lasting > max_rally_sec (default 25 s) is force-
      closed — very long single-rally spans almost certainly contain missed Points.
  18. OCR crop height gate: OCR is only attempted when the player bbox height is
      >= ocr_min_player_height_px (default 100 px) to prevent bad reads on tiny crops.
  19. Output filenames use only the original video stem — no split/holdout prefix.

Improvements applied (v6) — client feedback (Fabio) event sync + jerseys:
  20. Debug overlay: ball conf, tracker state, rally_active, touch_index, dead-ball timer.
  21. Ball tracker reset on Point; adaptive YOLO conf on miss streak; trajectory-aware
      multi-candidate ball selection.
  22. Dual-signal contacts (proximity + trajectory inflection); short event segments;
      upper-body association for non-serve touches; adaptive contact cooldown.
  23. RallySequenceNormalizer: Serve→Reception→Set→Attack grammar (kill = Attack).
  24. Dead-ball point detection + inter-rally lockout (fix late points / false restarts).
  25. Jersey-at-touch OCR burst, roster filter, CSV conf threshold (blank > wrong).

Improvements applied (v7) — 5-min holdout review (Fabio still failing):
  26. Serve grace + min gap after last touch before Point; ignore floor-band during serve.
  27. Point timestamp never earlier than last touch end (timeline enforce pass).
  28. Stronger serve-only rally start gate; longer Kalman coast for in-rally contacts.
  29. Jersey dominance dampening (blank over #6 flood); single CSV by default.
  30. Robust AVI→MP4(+audio) conversion with no-audio fallback; delete leftover AVI.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

EVENT_NAMES = [
    "Serve",
    "Reception",
    "Set",
    "Attack",
    "Block",
    "Dig",
    "Rally",
    "Point",
]

TOUCH_EVENT_NAMES = {"Serve", "Reception", "Set", "Attack", "Block", "Dig"}

# Fabio review priority rallies (Villadoro client feedback video).
FABIO_REVIEW_RALLIES = [1, 2, 4, 6, 7, 10]

COCO_PERSON = 0
COCO_SPORTS_BALL = 32
JERSEY_RE = re.compile(r"^[0-9]{1,2}$")

_paddle_ocr = None


def format_timestamp(seconds: float) -> str:
    """Return HH:MM:SS string (no milliseconds – easier for humans to read)."""
    s = max(0, int(round(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_ms(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_part, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=project_root_from_here)
    player_weights: str = "yolo11n.pt"
    ball_weights: Optional[str] = None
    finetuned_player: Optional[Path] = None
    finetuned_ball: Optional[Path] = None
    jersey_detector_weights: Optional[Path] = None
    prefer_finetuned_player: bool = True
    prefer_finetuned_ball: bool = True
    image_size: int = 640
    ball_image_size: int = 1280
    conf_player: float = 0.35
    conf_ball: float = 0.12
    conf_ball_fallback: float = 0.06
    infer_stride: int = 1
    player_infer_stride: int = 1
    tracker: str = "botsort.yaml"
    ocr_interval_sec: float = 1.0
    device: Any = 0
    event_segment_sec: float = 0.35
    # FIX: was 250 → blocked every touch after first one; lowered to 30 so that
    # Set / Attack / Block / Dig can fire during a multi-touch rally.
    min_ball_speed: float = 25.0
    contact_cooldown_sec: float = 0.55
    contact_min_frames: int = 2
    contact_dist_px: float = 160.0
    rally_gap_sec: float = 3.5
    use_paddle_ocr: bool = True
    ocr_on_cpu: bool = False
    use_motion_fallback: bool = True
    ball_max_jump_px: float = 120.0
    min_ball_conf_contact: float = 0.30
    ball_confirm_frames: int = 2
    jersey_vote_min: int = 2
    save_jersey_crops: bool = True
    crops_dir: Optional[Path] = None
    net_y_ratio: float = 0.42
    court_margin_x: float = 0.05
    court_margin_y_top: float = 0.08
    court_margin_y_bot: float = 0.05

    # ── v3/v4: Jersey identity improvements ──────────────────────────────────
    # Valid volleyball jersey number range [1, max_jersey_number].
    # Numbers outside this range are rejected as OCR misreads.
    # Lowered from 99 to 25 — typical club volleyball rosters rarely exceed 25.
    max_jersey_number: int = 25

    # Minimum bbox area (px²) for a player crop to be considered "large enough"
    # for reliable OCR. Raised from 3000 to 5000 (~70×70 px minimum).
    ocr_min_bbox_area: int = 5000

    # Laplacian variance threshold for blur detection.
    # Raised from 40 to 60 — stricter sharpness requirement reduces bad votes.
    ocr_blur_threshold: float = 60.0

    # Brightness window [min, max] in mean pixel value (0-255).
    # Too dark or too bright frames give poor OCR results.
    ocr_brightness_min: float = 40.0
    ocr_brightness_max: float = 230.0

    # When True, confirmed tracks (jersey already locked) completely skip OCR.
    # This avoids wasting GPU time on already-identified players.
    ocr_skip_confirmed: bool = True

    # Minimum OCR interval in seconds for unconfirmed tracks.
    # Good-quality frames can still trigger OCR sooner than ocr_interval_sec.
    ocr_min_interval_sec: float = 0.5

    # ── v4: Tighter confirmation gate ──────────────────────────────────────
    # Minimum number of agreeing votes before a track identity is locked.
    # Raised from 2 to 3 to prevent false confirmation on single fleeting reads.
    jersey_vote_min: int = 3

    # Minimum *average* per-vote confidence required to confirm a track.
    # A track accumulating 3 votes of confidence 0.36 is NOT confirmed.
    confirm_min_avg_conf: float = 0.55

    # Minimum per-vote confidence to accept an OCR result into the vote pool.
    # Raised from 0.35 to 0.50 — low-confidence reads add noise more than signal.
    ocr_min_vote_conf: float = 0.50

    # Maximum spatial distance (px) between two track bboxes used during
    # post-process track clustering. Tracks whose bboxes are further apart
    # are NOT merged even if they share the same jersey number.
    # v6: tightened from 200 → 150 to reduce #6 identity floods.
    cluster_max_bbox_gap_px: float = 150.0

    # Maximum time gap (seconds) between the end of one track and the start of
    # another for them to be considered the same physical player during clustering.
    # v6: tightened from 8 → 5 s.
    cluster_max_time_gap_sec: float = 5.0

    # ── v5: Event logic correctness ────────────────────────────────────────
    # Maximum seconds a single rally may last before it is force-closed.
    # Rallies longer than this almost certainly missed a Point detection.
    max_rally_sec: float = 25.0

    # Minimum player bounding-box height (px) required before OCR is attempted.
    # Crops shorter than this are too small for reliable digit reading.
    ocr_min_player_height_px: int = 100

    # Maximum seconds after a Serve event during which the next contact is
    # retroactively relabelled Reception (first-ball pass, not Dig).
    serve_to_reception_window_sec: float = 5.0

    # ── v6: Debug / ball / contact / phase / jersey ─────────────────────────
    debug_overlay: bool = False

    # After this many consecutive YOLO ball misses, temporarily use conf_ball_fallback
    # inside the court ROI (adaptive confidence).
    ball_miss_adaptive_frames: int = 4

    # Fixed display length of a touch event segment (seconds). Prevents Serve
    # from spanning the entire rally when mid-rally contacts are missed.
    event_display_sec: float = 0.40

    # Trajectory inflection: relative speed drop + direction change that marks a touch.
    inflection_speed_drop_ratio: float = 0.45
    inflection_min_prev_speed: float = 50.0
    inflection_dir_change_deg: float = 35.0

    # Adaptive contact cooldown after high-speed phases (Attack/Block).
    contact_cooldown_fast_sec: float = 0.45
    contact_fast_speed_threshold: float = 120.0

    # Dead-ball point detection (shorter than rally_gap_sec).
    # v7: slightly longer so brief ball dropouts after Serve do not kill the rally.
    point_dead_ball_sec: float = 1.4
    # v8: a rally may only START after this many YOLO-confirmed ball detections
    # inside rally_start_window_sec (motion/Hough fallback and Kalman-predicted
    # balls never start a rally).
    rally_start_min_ball_frames: int = 3
    rally_start_window_sec: float = 1.0
    # v8: a rally with fewer YOLO ball detections than this is a false rally
    # (huddle, warm-up, static ball lookalike) and is DISCARDED, not exported.
    min_rally_ball_frames: int = 5
    # v8: YOLO detections may count as contacts at a lower confidence than
    # motion-fallback balls (tune against ground truth).
    min_ball_conf_yolo_contact: float = 0.25
    # v8: a stationary ball only counts as 'dead' if it was moving fast this
    # recently (landing); stops static false positives from ending rallies.
    dead_requires_recent_fast_sec: float = 2.0
    dead_fast_speed_px: float = 60.0
    point_dead_speed_px: float = 25.0
    # Hard lockout after Point before a new Rally may start (unless serve-like).
    inter_rally_lockout_sec: float = 3.0
    # Floor band (bottom fraction of frame) used as out-of-play / bounce cue.
    floor_band_ratio: float = 0.90
    # After Serve, do not start dead-ball / floor Point until this many seconds
    # (gives Reception a chance — Fabio's main missed in-rally events).
    serve_grace_sec: float = 2.2
    # Point time must be at least this far after the last touch start.
    min_point_after_touch_sec: float = 0.35
    # Mid-rally Kalman coast frames (longer = recover Reception/Set when YOLO blinks).
    ball_predict_frames: int = 10
    # Slightly looser contact association for in-rally touches.
    contact_dist_mid_rally_px: float = 180.0

    # Jersey CSV policy: blank rather than publish low-confidence numbers.
    jersey_export_min_conf: float = 0.50
    # If one jersey accounts for more than this fraction of filled touch rows, blank
    # lower-confidence copies of that number (stops #6 flooding unrelated players).
    jersey_dominance_max_frac: float = 0.35
    # Optional roster of valid jersey numbers for this match (empty = use max_jersey_number).
    roster_numbers: Optional[Set[int]] = None


class CourtROI:
    """Exclude sidelines/ads from ball search."""

    def __init__(self, w: int, h: int, cfg: PipelineConfig):
        self.x1 = int(w * cfg.court_margin_x)
        self.x2 = int(w * (1.0 - cfg.court_margin_x))
        self.y1 = int(h * cfg.court_margin_y_top)
        self.y2 = int(h * (1.0 - cfg.court_margin_y_bot))

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class JerseyReader:
    """PaddleOCR / EasyOCR on upper-torso crops with temporal voting.

    v3 improvements:
    - Jersey number range filter: rejects numbers outside [1, max_jersey_number]
      before adding to votes, eliminating scoreboard/logo OCR misreads.
    - Best-frame selection: frame_quality_score() evaluates each crop before OCR.
      Only frames above a quality threshold run OCR, reducing bad votes.
    - Confirmed track short-circuit: once a track is confirmed, OCR is skipped
      entirely (controlled by ocr_skip_confirmed in PipelineConfig).

    v4 improvements:
    - Tighter confirmation: requires vote_min (now 3) votes AND avg confidence
      >= confirm_min_avg_conf before locking a track identity.
    - Anti-spam voting: ≥3 consecutive identical reads for the same track count
      as a single vote to prevent stationary-player flooding.
    - Upper-body-only crop: torso restricted to top 55 % of bbox before region
      variants are applied, removing leg/shoe area containing sponsor numbers.
    - Raised per-vote confidence floor (ocr_min_vote_conf, default 0.50).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        vote_min: int = 3,
        crops_dir: Optional[Path] = None,
        max_jersey_number: int = 25,
        blur_threshold: float = 60.0,
        brightness_min: float = 40.0,
        brightness_max: float = 230.0,
        min_bbox_area: int = 5000,
        confirm_min_avg_conf: float = 0.55,
        ocr_min_vote_conf: float = 0.50,
        roster_numbers: Optional[Set[int]] = None,
    ):
        self.use_gpu = use_gpu
        self.vote_min = vote_min
        self.crops_dir = Path(crops_dir) if crops_dir else None
        self.max_jersey_number = max_jersey_number
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.min_bbox_area = min_bbox_area
        self.confirm_min_avg_conf = confirm_min_avg_conf
        self.ocr_min_vote_conf = ocr_min_vote_conf
        self.roster_numbers = set(roster_numbers) if roster_numbers else None
        self._votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._vote_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Anti-spam: tracks last voted number and consecutive-frame counter
        self._last_voted_num: Dict[int, str] = {}
        self._consecutive_same: Dict[int, int] = defaultdict(int)
        self._confirmed: Dict[int, Tuple[str, float]] = {}
        if self.crops_dir:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

    def _get_ocr(self):
        global _paddle_ocr
        if _paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                _paddle_ocr = PaddleOCR(
                    use_angle_cls=False, lang="en", show_log=False,
                    use_gpu=self.use_gpu, det=True, rec=True,
                )
                _paddle_ocr._backend = "paddle"
            except Exception:
                import easyocr
                _paddle_ocr = easyocr.Reader(["en"], gpu=self.use_gpu)
                _paddle_ocr._backend = "easyocr"
        return _paddle_ocr

    @staticmethod
    def _torso_crops(bgr: np.ndarray) -> List[np.ndarray]:
        """Return a list of upper-body crop variants suitable for OCR.

        v4 change: the search area is first restricted to the top 55 % of the
        player bbox (jersey is on the torso, NOT on legs/shoes which often carry
        sponsor numbers that cause false reads).  Region variants are then applied
        inside that restricted strip.
        """
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        if h < 18 or w < 12:
            return []
        # If a dedicated jersey-number detector has already produced a tight
        # crop, do not crop it again; OCR needs the full number patch.
        aspect = w / max(h, 1)
        if h < 80 or aspect >= 0.8:
            return [bgr]

        # ── v4: Restrict to upper body (top 55 %) before region variants ──
        upper_h = int(h * 0.55)
        upper = bgr[0:upper_h, :]
        uh, uw = upper.shape[:2]
        if uh < 10:
            upper = bgr  # fallback: too small after crop, use full
            uh, uw = h, w

        regions = [
            upper[int(uh * 0.05): int(uh * 0.95), int(uw * 0.08): int(uw * 0.92)],
            upper[int(uh * 0.10): int(uh * 0.90), int(uw * 0.15): int(uw * 0.85)],
            upper[int(uh * 0.00): int(uh * 1.00), int(uw * 0.05): int(uw * 0.95)],
            upper[int(uh * 0.15): int(uh * 0.85), int(uw * 0.00): int(uw * 1.00)],
        ]
        out = []
        for torso in regions:
            if torso.size == 0:
                continue
            scale = max(1.0, 200 / max(torso.shape[0], 1))
            if scale > 1.05:
                torso = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            out.append(torso)
        return out

    @staticmethod
    def _preprocess(gray: np.ndarray) -> np.ndarray:
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        return gray

    def frame_quality_score(self, bgr_crop: np.ndarray) -> float:
        """Return a quality score [0.0, 1.0] for a player crop.

        Combines three signals:
          - bbox_area_ok  : crop is large enough for reliable OCR
          - sharpness     : Laplacian variance (high = sharp, low = blurry)
          - brightness_ok : mean pixel value within the usable range

        Returns 0.0 for crops that are definitely too poor for OCR.
        Returns > 0.5 for good-quality frames worth processing.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return 0.0
        h, w = bgr_crop.shape[:2]
        area = h * w

        # Gate 1: too small → skip entirely
        if area < self.min_bbox_area:
            return 0.0

        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        # Gate 2: sharpness via Laplacian variance
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < self.blur_threshold:
            return 0.0

        # Gate 3: brightness
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.brightness_min or mean_brightness > self.brightness_max:
            return 0.0

        # Combine into [0, 1] score
        # Sharpness: log-scaled, capped at 500
        sharpness_score = min(1.0, blur_var / 500.0)
        # Size: larger bbox → better, capped at 20000 px²
        size_score = min(1.0, area / 20000.0)
        # Brightness proximity to ideal (130)
        brightness_score = 1.0 - abs(mean_brightness - 130.0) / 130.0

        return (sharpness_score * 0.50 + size_score * 0.30 + brightness_score * 0.20)

    def _is_valid_jersey_number(self, number_str: str) -> bool:
        """Return True only if number_str is a plausible volleyball jersey number.

        Rules:
        - Must be 1 or 2 digits (already enforced by JERSEY_RE upstream)
        - Must not be "0" or "00" (no player wears zero)
        - Must be in [1, max_jersey_number]
        - If roster_numbers is set, must be in that set
        """
        if not number_str or number_str in ("0", "00"):
            return False
        try:
            val = int(number_str.lstrip("0") or "0")
        except ValueError:
            return False
        if not (1 <= val <= self.max_jersey_number):
            return False
        if self.roster_numbers is not None and val not in self.roster_numbers:
            return False
        return True

    def read_once(self, bgr_crop: np.ndarray) -> Tuple[Optional[str], float]:
        crops = self._torso_crops(bgr_crop)
        if not crops:
            return None, 0.0
        best_num, best_conf = None, 0.0
        try:
            ocr = self._get_ocr()
        except Exception:
            return None, 0.0

        backend = getattr(ocr, "_backend", "paddle")
        for torso in crops:
            gray = self._preprocess(cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY))
            try:
                if backend == "easyocr":
                    results = ocr.readtext(gray, detail=1, paragraph=False, allowlist="0123456789")
                    for (_bbox, text_raw, conf) in results:
                        text = re.sub(r"\D", "", str(text_raw).strip())
                        if not JERSEY_RE.match(text) or conf < 0.30:
                            continue
                        norm = text.lstrip("0") or text
                        # ── v3: range filter ──────────────────────────────
                        if not self._is_valid_jersey_number(norm):
                            continue
                        if conf > best_conf:
                            best_num, best_conf = norm, float(conf)
                else:
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    lines = ocr.ocr(bgr, cls=False)
                    if not lines:
                        continue
                    for line in lines:
                        if not line:
                            continue
                        for item in line:
                            if not item or len(item) < 2:
                                continue
                            text_raw, conf = item[1][0], float(item[1][1])
                            text = re.sub(r"\D", "", str(text_raw).strip())
                            if not JERSEY_RE.match(text) or conf < 0.35:
                                continue
                            norm = text.lstrip("0") or text
                            # ── v3: range filter ──────────────────────────
                            if not self._is_valid_jersey_number(norm):
                                continue
                            if conf > best_conf:
                                best_num, best_conf = norm, conf
            except Exception:
                continue
        return best_num, best_conf

    def update_track(self, track_id: int, bgr_crop: np.ndarray, frame_idx: int = 0) -> Tuple[Optional[str], float]:
        """Accumulate OCR votes for a track and return the current best identity.

        v4 changes:
        - Minimum per-vote confidence raised to ocr_min_vote_conf (default 0.50).
        - Anti-spam: ≥3 consecutive frames voting the same number count as 1 vote.
        - Confirmation requires vote_min agreeing votes AND avg confidence ≥
          confirm_min_avg_conf (both thresholds configurable via PipelineConfig).
        """
        num, conf = self.read_once(bgr_crop)

        if num and conf >= self.ocr_min_vote_conf:
            # ── v4: Anti-spam consecutive-frame gate ──────────────────────
            prev_num = self._last_voted_num.get(track_id)
            if prev_num == num:
                self._consecutive_same[track_id] += 1
            else:
                self._consecutive_same[track_id] = 1
                self._last_voted_num[track_id] = num

            # Allow the first 2 consecutive frames freely; only suppress
            # frame 3+ to avoid drowning out a correct minority number.
            if self._consecutive_same[track_id] <= 2:
                self._votes[track_id][num] += conf
                self._vote_counts[track_id][num] += 1
                if self.crops_dir and conf >= 0.50:
                    crop_path = self.crops_dir / f"tid{track_id}_f{frame_idx}_{num}.jpg"
                    cv2.imwrite(str(crop_path), bgr_crop)

        if not self._votes[track_id]:
            return num if num else None, conf

        best_num, score = max(self._votes[track_id].items(), key=lambda kv: kv[1])
        votes_for_best = self._vote_counts[track_id].get(best_num, 0)

        # ── v4: Average-confidence gate ────────────────────────────────────
        avg_conf = score / max(votes_for_best, 1)
        norm_conf = min(1.0, score / max(self.vote_min, 1))

        ready = (
            votes_for_best >= self.vote_min
            and avg_conf >= self.confirm_min_avg_conf
        )
        if ready:
            self._confirmed[track_id] = (best_num, norm_conf)
        return best_num, norm_conf

    def is_confirmed(self, track_id: int) -> bool:
        """Return True if this track_id already has a locked jersey identity."""
        return track_id in self._confirmed

    def get_all_confirmed(self) -> Dict[int, Tuple[str, float]]:
        """Return every (track_id -> (jersey, conf)) that reached confirmation."""
        return dict(self._confirmed)


class JerseyRegionDetector:
    """Optional fine-tuned YOLO detector for printed jersey-number regions."""

    def __init__(self, weights: Optional[Path], device: Any, image_size: int):
        self.model: Optional[YOLO] = None
        self.device = device
        self.image_size = image_size
        if weights and Path(weights).exists():
            self.model = YOLO(str(weights))

    def crops(self, player_crop: np.ndarray) -> List[np.ndarray]:
        if self.model is None or player_crop is None or player_crop.size == 0:
            return []
        try:
            result = self.model.predict(player_crop, conf=0.35, verbose=False,
                                        device=self.device, imgsz=self.image_size)[0]
        except Exception:
            return []
        if result.boxes is None:
            return []
        h, w = player_crop.shape[:2]
        regions: List[np.ndarray] = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            region = player_crop[y1:y2, x1:x2]
            if region.shape[0] >= 10 and region.shape[1] >= 8:
                regions.append(region)
        return regions


class BallKalmanTracker:
    """Smooth ball position; reject impossible jumps."""

    def __init__(self, max_jump_px: float = 120.0):
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.vx = 0.0
        self.vy = 0.0
        self.conf = 0.0
        self.missed = 0
        self.max_jump_px = max_jump_px
        self.history: Deque[Tuple[float, float, float, float]] = deque(maxlen=30)
        self.consecutive_updates = 0
        self.max_prediction_frames = 6
        self.last_measured: Optional[Tuple[float, float, float]] = None
        self.last_predicted: Optional[Tuple[float, float, float]] = None

    def reset(self) -> None:
        """Clear track state after Point / dead ball to avoid walk-back hallucinations."""
        self.x = self.y = None
        self.vx = self.vy = self.conf = 0.0
        self.missed = 0
        self.consecutive_updates = 0
        self.history.clear()
        self.last_measured = None
        self.last_predicted = None

    def predict(self) -> Optional[Tuple[float, float, float]]:
        if self.x is None:
            return None
        if self.missed >= self.max_prediction_frames:
            self.x = self.y = None
            self.vx = self.vy = self.conf = 0.0
            self.consecutive_updates = 0
            self.last_predicted = None
            return None
        self.x += self.vx
        self.y += self.vy
        self.missed += 1
        self.consecutive_updates = 0
        decay = max(0.10, self.conf * (0.90 ** self.missed))
        self.last_predicted = (self.x, self.y, decay)
        return self.x, self.y, decay

    def update(self, cx: float, cy: float, conf: float, t: float = 0.0) -> bool:
        if self.x is not None:
            jump = ((cx - self.x) ** 2 + (cy - self.y) ** 2) ** 0.5
            if jump > self.max_jump_px and self.missed < 3:
                self.consecutive_updates = 0
                return False
        if self.x is None or self.missed >= 3:
            self.x, self.y = cx, cy
            self.vx = self.vy = 0.0
        else:
            self.vx = 0.50 * (cx - self.x) + 0.50 * self.vx
            self.vy = 0.50 * (cy - self.y) + 0.50 * self.vy
            self.x = 0.70 * cx + 0.30 * self.x
            self.y = 0.70 * cy + 0.30 * self.y
        self.conf = conf
        self.missed = 0
        self.consecutive_updates += 1
        self.history.append((t, self.x, self.y, conf))
        self.last_measured = (cx, cy, conf)
        return True

    def is_confirmed(self, min_frames: int) -> bool:
        return self.consecutive_updates >= max(1, min_frames)

    def speed_px_s(self) -> float:
        if len(self.history) < 3:
            return 0.0
        t0, x0, y0, _ = self.history[0]
        t1, x1, y1, _ = self.history[-1]
        dt = max(1e-3, t1 - t0)
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt)


@dataclass
class OpenSegment:
    event: str
    frame_start: int
    time_start: float
    player_track_id: Optional[int] = None
    jersey_number: str = ""
    jersey_conf: float = 0.0
    ball_speed_peak: float = 0.0
    event_conf: float = 0.0
    rally_id: int = 0
    touch_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class RallyEventEngine:
    """
    Multi-frame validated touches, rally segmentation, short event windows (v6).

    v6 changes vs v5:
      • Short fixed display segments (event_display_sec) instead of open-until-next.
      • Dual-signal contacts: proximity OR trajectory inflection.
      • Dead-ball point detection + inter-rally lockout.
      • Adaptive cooldown after high-speed Attack/Block.
      • Upper-body preference for non-serve player association.
    """

    def __init__(self, fps: float, cfg: PipelineConfig):
        self.fps = fps or 25.0
        self.cfg = cfg
        self.ball_hist: Deque[Tuple[float, float, float, float]] = deque(maxlen=60)
        self.last_contact_t = -1e9
        self.rally_active = False
        self.rally_id = 0
        self.touch_index = 0
        self.last_ball_t: Optional[float] = None
        self.last_valid_ball_t: Optional[float] = None
        self.open: Optional[OpenSegment] = None
        self.closed: List[dict] = []
        self.net_y: Optional[float] = None
        self._contact_frames = 0
        self._contact_tid: Optional[int] = None
        self._contact_jersey = ""
        self._contact_jconf = 0.0
        self._last_event_name = ""
        self._event_id = 0
        self._last_point_t: Optional[float] = None
        self._needs_ball_reset = False
        self._dead_ball_start_t: Optional[float] = None
        self._last_speed = 0.0
        self._last_touch_t: Optional[float] = None
        self._serve_t: Optional[float] = None
        self._ball_was_elevated = False
        # v8 state
        self._yolo_times: Deque[float] = deque(maxlen=64)
        self._rally_ball_frames = 0
        self._last_fast_t: Optional[float] = None
        self._last_yolo_t: Optional[float] = None
        self.discarded_rallies = 0
        self.debug_state: Dict[str, Any] = {}

    def _ball_ok(self, ball, src: str = "yolo") -> bool:
        """v8: confidence gate depends on where the ball came from."""
        if ball is None:
            return False
        thr = (
            self.cfg.min_ball_conf_yolo_contact
            if src == "yolo" else self.cfg.min_ball_conf_contact
        )
        return ball[2] >= thr

    def _start_ball_ok(self, t: float) -> bool:
        """v8: enough recent YOLO-confirmed ball frames to open a rally."""
        w = self.cfg.rally_start_window_sec
        n = sum(1 for x in self._yolo_times if t - x <= w)
        return n >= self.cfg.rally_start_min_ball_frames

    def _discard_current_rally(self, reason: str) -> None:
        """v8: remove every row of a false rally; keep ids contiguous."""
        rid = self.rally_id
        kept = [r for r in self.closed if r.get("rally_id") != rid]
        removed = len(self.closed) - len(kept)
        self.closed[:] = kept
        self._event_id = max(0, self._event_id - removed)
        self.rally_id = max(0, rid - 1)
        self.open = None
        self.discarded_rallies += 1
        print(f"  [v8] discarded false rally #{rid} ({reason}, {removed} rows)")

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def consume_ball_reset_flag(self) -> bool:
        flag = self._needs_ball_reset
        self._needs_ball_reset = False
        return flag

    def _cooldown_needed(self, speed: float) -> float:
        if speed >= self.cfg.contact_fast_speed_threshold or self._last_event_name in (
            "Attack", "Block",
        ):
            return self.cfg.contact_cooldown_fast_sec
        return self.cfg.contact_cooldown_sec

    def _cooldown_ok(self, t: float, speed: float = 0.0) -> bool:
        return (t - self.last_contact_t) >= self._cooldown_needed(speed)

    def _close_open(self, frame_idx: int, t: float, force_end: Optional[float] = None):
        if self.open is None:
            return
        if force_end is not None:
            end_t = max(force_end, self.open.time_start + 0.05)
        elif self.open.event in TOUCH_EVENT_NAMES:
            end_t = self.open.time_start + self.cfg.event_display_sec
        else:
            end_t = max(t, self.open.time_start + 0.05)
        # Never let a touch segment extend past the closing time
        end_t = min(end_t, max(t, self.open.time_start + 0.05)) if force_end is None and self.open.event not in TOUCH_EVENT_NAMES else end_t
        if self.open.event in TOUCH_EVENT_NAMES:
            end_t = self.open.time_start + self.cfg.event_display_sec
        row = {
            "event_id": self._next_event_id(),
            "event": self.open.event,
            "time_start_sec": round(self.open.time_start, 3),
            "time_end_sec": round(end_t, 3),
            "timestamp_start": format_timestamp_ms(self.open.time_start),
            "timestamp_end": format_timestamp_ms(end_t),
            "timestamp": format_timestamp(self.open.time_start),
            "frame_start": self.open.frame_start,
            "frame_end": int(frame_idx),
            "player_track_id": self.open.player_track_id or "",
            "jersey_number": self.open.jersey_number,
            "jersey_conf": round(self.open.jersey_conf, 3) if self.open.jersey_number else "",
            "event_conf": round(self.open.event_conf, 3),
            "ball_speed_peak": round(self.open.ball_speed_peak, 1),
            "rally_id": self.open.rally_id,
            "touch_index": self.open.touch_index,
            "event_status": "confirmed",
        }
        row.update(self.open.extra)
        self.closed.append(row)
        self.open = None

    def _emit_instant(
        self,
        name: str,
        frame_idx: int,
        t: float,
        tid: Optional[int],
        jersey: str,
        jconf: float,
        speed: float,
        event_conf: float = 0.8,
        extra: Optional[dict] = None,
        duration: Optional[float] = None,
    ):
        """Close any open segment, then emit a short fixed-length event."""
        self._close_open(frame_idx, t)
        dur = duration if duration is not None else (
            self.cfg.event_display_sec if name in TOUCH_EVENT_NAMES else 0.05
        )
        end_t = t + dur
        row = {
            "event_id": self._next_event_id(),
            "event": name,
            "time_start_sec": round(t, 3),
            "time_end_sec": round(end_t, 3),
            "timestamp_start": format_timestamp_ms(t),
            "timestamp_end": format_timestamp_ms(end_t),
            "timestamp": format_timestamp(t),
            "frame_start": frame_idx,
            "frame_end": int(frame_idx + max(1, int(dur * self.fps))),
            "player_track_id": tid or "",
            "jersey_number": jersey or "",
            "jersey_conf": round(jconf, 3) if jersey else "",
            "event_conf": round(event_conf, 3),
            "ball_speed_peak": round(speed, 1),
            "rally_id": self.rally_id,
            "touch_index": self.touch_index,
            "event_status": "confirmed",
        }
        if extra:
            row.update(extra)
        self.closed.append(row)
        self._last_event_name = name

    def _end_rally(
        self,
        frame_idx: int,
        t: float,
        speed: float,
        reason: str,
        point_time: Optional[float] = None,
    ):
        """v8: discards false rallies and back-dates the Point to when play stopped."""
        if not self.rally_active:
            return
        self._close_open(frame_idx, t)

        def _reset_state(last_point_t: float):
            self.rally_active = False
            self.touch_index = 0
            self._contact_frames = 0
            self._contact_tid = None
            self._last_point_t = last_point_t
            self._needs_ball_reset = True
            self._dead_ball_start_t = None
            self._last_touch_t = None
            self._serve_t = None
            self._ball_was_elevated = False
            self._rally_ball_frames = 0
            self._last_fast_t = None
            self._yolo_times.clear()
            self.ball_hist.clear()

        # False rally: the ball was (almost) never really seen.
        if self._rally_ball_frames < self.cfg.min_rally_ball_frames:
            self._discard_current_rally(f"{reason}; ball_frames={self._rally_ball_frames}")
            _reset_state(t)  # keep lockout so the same junk cannot re-trigger
            return

        # Back-date: Point = when play actually stopped, not when we decided.
        point_t = t if point_time is None else min(point_time, t)
        if self._last_touch_t is not None:
            min_t = self._last_touch_t + max(
                self.cfg.min_point_after_touch_sec,
                self.cfg.event_display_sec,
            )
            point_t = max(point_t, min_t)
        self.touch_index += 1
        self._emit_instant(
            "Point", int(point_t * self.fps), point_t, None, "", 0.0, speed,
            event_conf=1.0,
            extra={"detail": reason, "ball_frames": self._rally_ball_frames},
            duration=0.05,
        )
        _reset_state(max(t, point_t))  # lockout is measured from the decision time

    def _classify_touch(
        self,
        speed: float,
        vx: float,
        vy: float,
        ball_y: float,
        player_y: float,
        frame_h: float,
        is_rally_start: bool,
    ) -> Tuple[str, float]:
        net = self.net_y if self.net_y is not None else frame_h * 0.42
        ball_rel = ball_y / max(frame_h, 1)
        net_dist = abs(ball_y - net) / max(frame_h, 1)
        near_net = net_dist < 0.15

        if is_rally_start:
            if ball_rel > 0.62 or speed > 150 or vy < -60:
                return "Serve", 0.85
            return "Reception", 0.75

        if near_net and (abs(vy) > 60 or speed > 130):
            return "Block", 0.80

        if vy > 60 and speed > 100:
            return "Attack", 0.82
        if speed > 200 and ball_rel > 0.35:
            return "Attack", 0.75

        if near_net and vy < 0 and speed < 160:
            return "Set", 0.78
        if abs(vy) < 60 and speed < 140 and net_dist < 0.25:
            return "Set", 0.72

        if ball_rel > 0.55 and speed < 200:
            return "Dig", 0.74
        if ball_rel > 0.45 and speed < 120:
            return "Dig", 0.70

        if self.touch_index <= 1:
            return "Reception", 0.60
        if self.touch_index == 2:
            return "Set", 0.55
        return "Attack" if speed > 80 else "Dig", 0.55

    def _looks_like_serve(
        self,
        speed: float,
        vy: float,
        ball_y: float,
        player_y: float,
        frame_h: float,
    ) -> bool:
        """v7: stricter serve gate — deep player/ball + outbound/up speed."""
        ball_rel = ball_y / max(frame_h, 1)
        player_rel = player_y / max(frame_h, 1)
        deep = player_rel > 0.58 or ball_rel > 0.60
        serve_motion = speed > 100 or vy < -50
        return deep and serve_motion

    def _nearest_player(
        self,
        bx: float,
        by: float,
        player_boxes: Sequence[Tuple[int, int, int, int, int, str, float]],
        prefer_upper_body: bool = True,
    ) -> Optional[Tuple[int, str, float, float, float]]:
        best = None
        best_d = 1e18
        for (x1, y1, x2, y2, tid, jersey, jconf) in player_boxes:
            pad = self.cfg.contact_dist_px
            if not (x1 - pad <= bx <= x2 + pad and y1 - pad <= by <= y2 + pad):
                continue
            cx = (x1 + x2) / 2.0
            # Prefer upper-body (hands) for association on non-serve touches
            if prefer_upper_body:
                cy = y1 + 0.35 * (y2 - y1)
            else:
                cy = (y1 + y2) / 2.0
            d = (cx - bx) ** 2 + (cy - by) ** 2
            if d < best_d:
                best_d = d
                best = (tid, jersey, jconf, cy, float(d) ** 0.5)
        return best

    def _trajectory_inflection(self) -> bool:
        """Detect a contact via speed drop + direction change in ball history."""
        if len(self.ball_hist) < 4:
            return False
        pts = list(self.ball_hist)[-4:]
        speeds = []
        dirs = []
        for i in range(1, len(pts)):
            t0, x0, y0, _ = pts[i - 1]
            t1, x1, y1, _ = pts[i]
            dt = max(1e-3, t1 - t0)
            vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
            speeds.append((vx * vx + vy * vy) ** 0.5)
            dirs.append(math.atan2(vy, vx))
        if len(speeds) < 2:
            return False
        prev_speed = max(speeds[:-1])
        cur_speed = speeds[-1]
        if prev_speed < self.cfg.inflection_min_prev_speed:
            return False
        drop = (prev_speed - cur_speed) / max(prev_speed, 1e-3)
        if drop < self.cfg.inflection_speed_drop_ratio:
            return False
        # Direction change between first and last segment
        d0, d1 = dirs[0], dirs[-1]
        delta = abs(math.degrees((d1 - d0 + math.pi) % (2 * math.pi) - math.pi))
        return delta >= self.cfg.inflection_dir_change_deg or drop >= 0.65

    def _check_point_boundary(
        self,
        frame_idx: int,
        t: float,
        ball: Optional[Tuple[float, float, float]],
        speed: float,
        frame_h: int,
        src: str = "yolo",
    ) -> Optional[str]:
        """v8: a MISSING ball is no longer a DEAD ball; Point is back-dated."""
        if not self.rally_active:
            return None

        rally_start_t = next(
            (r["time_start_sec"] for r in reversed(self.closed)
             if r["rally_id"] == self.rally_id and r["event"] == "Rally"),
            None,
        )
        if rally_start_t is not None and (t - rally_start_t) > self.cfg.max_rally_sec:
            self._end_rally(frame_idx, t, speed, "rally_timeout")
            return "Point"

        ball_valid = self._ball_ok(ball, src)
        visible_yolo = ball_valid and src == "yolo"
        on_floor = False
        if visible_yolo:
            ball_rel = ball[1] / max(frame_h, 1)
            if ball_rel < 0.70:
                self._ball_was_elevated = True
            floor_ok = self._ball_was_elevated or self.touch_index >= 2
            on_floor = floor_ok and ball_rel >= self.cfg.floor_band_ratio
            if speed >= self.cfg.dead_fast_speed_px:
                self._last_fast_t = t

        recently_fast = (
            self._last_fast_t is not None
            and (t - self._last_fast_t) <= self.cfg.dead_requires_recent_fast_sec
        )
        in_serve_grace = (
            self._serve_t is not None
            and (t - self._serve_t) < self.cfg.serve_grace_sec
            and self.touch_index <= 1
        )

        # Dead ball needs a VISIBLE ball that is on the floor, or stopped right
        # after moving fast (landing). Missing frames neither arm nor reset it.
        slow_visible = (
            visible_yolo and speed < self.cfg.point_dead_speed_px and recently_fast
        )
        arm_dead = on_floor if in_serve_grace else (slow_visible or on_floor)

        if self.touch_index >= 1:
            if arm_dead:
                if self._dead_ball_start_t is None:
                    self._dead_ball_start_t = t
                elif (t - self._dead_ball_start_t) >= self.cfg.point_dead_ball_sec:
                    self._end_rally(
                        frame_idx, t, speed, "dead_ball",
                        point_time=self._dead_ball_start_t,
                    )
                    return "Point"
            elif visible_yolo and speed >= self.cfg.point_dead_speed_px:
                self._dead_ball_start_t = None  # ball is moving again
        else:
            self._dead_ball_start_t = None

        # Ball no longer seen by YOLO: end only after a longer gap, Point
        # back-dated to the last time the ball was actually seen.
        gap_limit = self.cfg.rally_gap_sec
        if in_serve_grace:
            gap_limit = max(gap_limit, self.cfg.serve_grace_sec + 1.0)
        ref_t = self._last_yolo_t if self._last_yolo_t is not None else self.last_valid_ball_t
        if ref_t is not None and (t - ref_t) > gap_limit:
            self._end_rally(frame_idx, t, speed, "ball_lost", point_time=ref_t)
            return "Point"
        return None

    def update(
        self,
        frame_idx: int,
        ball: Optional[Tuple[float, float, float]],
        player_boxes: Sequence[Tuple[int, int, int, int, int, str, float]],
        frame_h: int,
        ball_src: str = "yolo",
    ) -> Optional[str]:
        t = frame_idx / self.fps
        vx, vy, speed = 0.0, 0.0, 0.0
        contact_candidates = 0

        if self._ball_ok(ball, ball_src):
            bx, by, bconf = ball
            self.ball_hist.append((t, bx, by, bconf))
            self.last_ball_t = t
            self.last_valid_ball_t = t
            if ball_src == "yolo":
                self._yolo_times.append(t)
                self._last_yolo_t = t
                if self.rally_active:
                    self._rally_ball_frames += 1
            if len(self.ball_hist) >= 2:
                (_t0, x0, y0, _c0) = self.ball_hist[-2]
                (t1, x1, y1, _c1) = self.ball_hist[-1]
                dt = max(1e-3, t1 - _t0)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                speed = (vx * vx + vy * vy) ** 0.5
            self._last_speed = speed

        point_ev = self._check_point_boundary(frame_idx, t, ball, speed, frame_h, ball_src)
        self.debug_state = {
            "rally_active": self.rally_active,
            "rally_id": self.rally_id,
            "touch_index": self.touch_index,
            "ball_conf": round(ball[2], 3) if ball else 0.0,
            "ball_speed": round(speed, 1),
            "sec_since_ball": round(t - self.last_valid_ball_t, 2) if self.last_valid_ball_t else None,
            "dead_ball_sec": (
                round(t - self._dead_ball_start_t, 2) if self._dead_ball_start_t else 0.0
            ),
            "lockout_remaining": max(
                0.0,
                self.cfg.inter_rally_lockout_sec - (t - self._last_point_t),
            ) if self._last_point_t is not None else 0.0,
            "contact_candidates": 0,
            "last_event": self._last_event_name,
            "ball_src": ball_src,
            "rally_ball_frames": self._rally_ball_frames,
        }
        if point_ev:
            return point_ev

        if not self._ball_ok(ball, ball_src):
            self._contact_frames = 0
            self._contact_tid = None
            return None

        bx, by, bconf = ball
        is_start = not self.rally_active
        prefer_upper = not is_start
        nearest = self._nearest_player(bx, by, player_boxes, prefer_upper_body=prefer_upper)
        inflection = self._trajectory_inflection()

        contact_dist = (
            self.cfg.contact_dist_px if is_start else self.cfg.contact_dist_mid_rally_px
        )
        proximity_ok = False
        tid, jersey, jconf, player_y, dist = None, "", 0.0, by, 1e9
        if nearest is not None:
            tid, jersey, jconf, player_y, dist = nearest
            contact_candidates = 1
            proximity_ok = dist <= contact_dist

        self.debug_state["contact_candidates"] = contact_candidates

        # Dual-signal: proximity (multi-frame) OR strong inflection near a player
        min_frames = self.cfg.contact_min_frames
        if proximity_ok and tid is not None:
            if self._contact_tid == tid:
                self._contact_frames += 1
            else:
                self._contact_tid = tid
                self._contact_frames = 1
                self._contact_jersey = jersey
                self._contact_jconf = jconf
        elif inflection and nearest is not None and dist <= contact_dist * 1.5:
            # Inflection contact: require only 1 frame of association
            self._contact_tid = tid
            self._contact_frames = min_frames
            self._contact_jersey = jersey
            self._contact_jconf = jconf
            min_frames = 1
        elif (
            self.rally_active
            and self.touch_index >= 1
            and nearest is not None
            and dist <= contact_dist * 1.15
            and speed >= self.cfg.min_ball_speed
        ):
            # v7: mid-rally proximity with motion — allow 1-frame contacts
            self._contact_tid = tid
            self._contact_frames = max(self._contact_frames, 1)
            self._contact_jersey = jersey or self._contact_jersey
            self._contact_jconf = jconf or self._contact_jconf
            min_frames = 1
        else:
            self._contact_frames = 0
            self._contact_tid = None
            return None

        if self._contact_frames < min_frames:
            return None
        if not self._cooldown_ok(t, speed):
            return None

        # Inter-rally lockout: only allow serve-like contacts to start a new rally.
        # v7: outside lockout, still require serve-like evidence (no walk-back rallies).
        if is_start:
            # v8: only YOLO-confirmed balls, seen several times, may open a rally.
            if ball_src != "yolo" or not self._start_ball_ok(t):
                return None
            if not self._looks_like_serve(speed, vy, by, player_y, float(frame_h)):
                if self._last_point_t is not None:
                    return None
                # First rally of the video: still prefer serve-like, but allow deep contact
                if (player_y / max(frame_h, 1)) < 0.50 and speed < 120:
                    return None
            if self._last_point_t is not None:
                since_point = t - self._last_point_t
                if since_point < self.cfg.inter_rally_lockout_sec:
                    if not self._looks_like_serve(speed, vy, by, player_y, float(frame_h)):
                        return None

        # Mid-rally: skip only when ball is nearly stationary AND no inflection
        if (
            speed < self.cfg.min_ball_speed
            and self.rally_active
            and self.touch_index > 1
            and not inflection
        ):
            return None

        self.last_contact_t = t
        jersey = self._contact_jersey or jersey
        jconf = self._contact_jconf or jconf

        if is_start:
            self.rally_id += 1
            self.touch_index = 0
            self.rally_active = True
            self._ball_was_elevated = False
            self._last_fast_t = None
            self._rally_ball_frames = sum(
                1 for x in self._yolo_times
                if t - x <= self.cfg.rally_start_window_sec
            )
            self._emit_instant(
                "Rally", frame_idx, t, tid, jersey, jconf, speed,
                event_conf=1.0, extra={"detail": "rally_start"}, duration=0.05,
            )

        self.touch_index += 1
        ev, ev_conf = self._classify_touch(
            speed, vx, vy, by, player_y, float(frame_h), is_start,
        )
        self._emit_instant(
            ev, frame_idx, t, tid, jersey, jconf, speed,
            event_conf=ev_conf,
            extra={"ball_conf": round(bconf, 3), "vx": round(vx, 1), "vy": round(vy, 1)},
        )
        self._last_touch_t = t
        if ev == "Serve":
            self._serve_t = t
        self._dead_ball_start_t = None
        self.debug_state["last_event"] = ev
        self.debug_state["touch_index"] = self.touch_index
        return ev

    def finalize(self, frame_idx: int):
        t = frame_idx / self.fps
        if self.rally_active:
            self._end_rally(frame_idx, t, 0.0, "video_end")
        else:
            self._close_open(frame_idx, t)

    def backfill_jerseys(self, jersey_cache: Dict[int, str], jersey_conf_cache: Dict[int, float]):
        """Post-process jersey join; blank low-confidence / dominant OCR floods (v7)."""
        min_conf = self.cfg.jersey_export_min_conf
        for row in self.closed:
            tid = row.get("player_track_id")
            if not tid:
                continue
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int in jersey_cache:
                conf = float(jersey_conf_cache.get(tid_int, 0.0))
                if conf >= min_conf:
                    row["jersey_number"] = jersey_cache[tid_int]
                    row["jersey_conf"] = round(conf, 3)
                elif not row.get("jersey_number"):
                    row["jersey_number"] = ""
                    row["jersey_conf"] = ""
            # Blank existing low-confidence jerseys
            existing = row.get("jersey_number")
            existing_conf = row.get("jersey_conf")
            if existing:
                try:
                    ec = float(existing_conf) if existing_conf not in ("", None) else 0.0
                except (TypeError, ValueError):
                    ec = 0.0
                if ec < min_conf:
                    row["jersey_number"] = ""
                    row["jersey_conf"] = ""

        # v7: dominance dampening — if one number floods touch rows, blank weaker copies
        touch_rows = [r for r in self.closed if r.get("event") in TOUCH_EVENT_NAMES]
        filled = []
        for r in touch_rows:
            jn = r.get("jersey_number")
            if jn in ("", None):
                continue
            filled.append(r)
        if len(filled) >= 6:
            counts: Dict[str, int] = defaultdict(int)
            for r in filled:
                counts[str(r["jersey_number"]).split(".")[0]] += 1
            max_frac = self.cfg.jersey_dominance_max_frac
            dominant = {
                num for num, c in counts.items()
                if c / max(len(filled), 1) > max_frac
            }
            if dominant:
                # Also count distinct tracks claiming each dominant number
                tracks_by_num: Dict[str, Set[str]] = defaultdict(set)
                for r in filled:
                    num = str(r["jersey_number"]).split(".")[0]
                    if num in dominant and r.get("player_track_id") not in ("", None):
                        tracks_by_num[num].add(str(r["player_track_id"]))
                blanked = 0
                for r in filled:
                    num = str(r["jersey_number"]).split(".")[0]
                    if num not in dominant:
                        continue
                    try:
                        ec = float(r.get("jersey_conf") or 0.0)
                    except (TypeError, ValueError):
                        ec = 0.0
                    # Many tracks → same number is almost always OCR flood (#6).
                    # Keep only very high-confidence rows; blank the rest.
                    many_tracks = len(tracks_by_num.get(num, ())) >= 3
                    keep_floor = 0.90 if many_tracks else max(min_conf + 0.20, 0.70)
                    if ec < keep_floor:
                        r["jersey_number"] = ""
                        r["jersey_conf"] = ""
                        blanked += 1
                if blanked:
                    print(
                        f"  [v7 jersey] Blanked {blanked} low-conf dominant "
                        f"jersey(s) {sorted(dominant)} (flood guard)"
                    )

    def apply_event_logic_fixes(self, serve_to_reception_window_sec: float = 5.0) -> int:
        """v5 post-processing: Reception/Dig, Serve fallback, Block cleanup."""
        fixes = 0
        by_rally: Dict[int, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.closed):
            by_rally[row.get("rally_id", 0)].append(idx)

        for _rally_id, indices in by_rally.items():
            rally_rows = [self.closed[i] for i in indices]

            for pos, row in enumerate(rally_rows):
                if row["event"] != "Serve":
                    continue
                serve_t = row["time_end_sec"]
                for j in range(pos + 1, len(rally_rows)):
                    nxt = rally_rows[j]
                    if nxt["event"] in ("Rally", "Point"):
                        break
                    if nxt["time_start_sec"] - serve_t > serve_to_reception_window_sec:
                        break
                    if nxt["event"] == "Dig":
                        self.closed[indices[j]]["event"] = "Reception"
                        self.closed[indices[j]]["event_conf"] = 0.75
                        fixes += 1
                        break

            rally_start_idx = next(
                (i for i, r in enumerate(rally_rows) if r["event"] == "Rally"), None
            )
            if rally_start_idx is not None:
                rally_t = rally_rows[rally_start_idx]["time_end_sec"]
                has_serve = any(
                    r["event"] == "Serve"
                    and r["time_start_sec"] - rally_t <= serve_to_reception_window_sec
                    for r in rally_rows
                )
                if not has_serve:
                    for j in range(rally_start_idx + 1, len(rally_rows)):
                        r = rally_rows[j]
                        if r["event"] in ("Rally", "Point"):
                            continue
                        if r["time_start_sec"] - rally_t <= serve_to_reception_window_sec:
                            self.closed[indices[j]]["event"] = "Serve"
                            self.closed[indices[j]]["event_conf"] = 0.80
                            fixes += 1
                            break

            for pos, row in enumerate(rally_rows):
                if row["event"] != "Block":
                    continue
                block_t = row["time_start_sec"]
                preceding_attack = any(
                    r["event"] == "Attack"
                    and 0 <= block_t - r["time_start_sec"] <= 2.0
                    for r in rally_rows[:pos]
                )
                if not preceding_attack:
                    self.closed[indices[pos]]["event"] = "Attack"
                    self.closed[indices[pos]]["event_conf"] = 0.72
                    fixes += 1

        if fixes:
            print(f"  [v5 event fixes] Applied {fixes} event relabelling correction(s)")
        return fixes


class RallySequenceNormalizer:
    """v6: per-rally grammar — Serve → Reception → Set → Attack (kill = Attack).

    Only relabels existing contacts; does not invent new events.
    """

    def __init__(self, serve_to_reception_window_sec: float = 5.0):
        self.window = serve_to_reception_window_sec

    def apply(self, closed: List[dict]) -> int:
        fixes = 0
        by_rally: Dict[int, List[int]] = defaultdict(list)
        for idx, row in enumerate(closed):
            by_rally[row.get("rally_id", 0)].append(idx)

        for _rid, indices in by_rally.items():
            rows = [closed[i] for i in indices]
            touch_idxs = [
                (j, indices[j])
                for j, r in enumerate(rows)
                if r["event"] in TOUCH_EVENT_NAMES
            ]
            if not touch_idxs:
                continue

            # Ensure first touch is Serve when possible
            first_j, first_gi = touch_idxs[0]
            if rows[first_j]["event"] != "Serve":
                # Ace-like: keep as-is if very high speed Attack immediately
                peak = float(rows[first_j].get("ball_speed_peak") or 0)
                if rows[first_j]["event"] == "Attack" and peak > 220:
                    pass
                else:
                    closed[first_gi]["event"] = "Serve"
                    closed[first_gi]["event_conf"] = 0.80
                    fixes += 1
                    rows[first_j] = closed[first_gi]

            # First post-serve touch → Reception (unless Block overpass)
            serve_pos = next(
                (j for j, r in enumerate(rows) if r["event"] == "Serve"), None
            )
            if serve_pos is not None:
                serve_t = rows[serve_pos]["time_end_sec"]
                for j in range(serve_pos + 1, len(rows)):
                    r = rows[j]
                    if r["event"] in ("Rally", "Point"):
                        break
                    if r["event"] not in TOUCH_EVENT_NAMES:
                        continue
                    if r["time_start_sec"] - serve_t > self.window:
                        break
                    if r["event"] == "Block":
                        break  # overpass edge case
                    if r["event"] != "Reception":
                        # v7.1: always relabel first post-Serve touch as Reception
                        # (Fabio: Serve then receive/set/kill — never Serve→Attack first).
                        closed[indices[j]]["event"] = "Reception"
                        closed[indices[j]]["event_conf"] = 0.75
                        fixes += 1
                        rows[j] = closed[indices[j]]
                    break

            # After Reception, near-net gentle / mid touch → Set
            for j, r in enumerate(rows):
                if r["event"] != "Reception":
                    continue
                for k in range(j + 1, len(rows)):
                    nxt = rows[k]
                    if nxt["event"] in ("Rally", "Point"):
                        break
                    if nxt["event"] not in TOUCH_EVENT_NAMES:
                        continue
                    if nxt["event"] in ("Attack", "Block", "Set"):
                        break
                    peak = float(nxt.get("ball_speed_peak") or 0)
                    if nxt["event"] in ("Dig",) and peak < 160:
                        closed[indices[k]]["event"] = "Set"
                        closed[indices[k]]["event_conf"] = 0.72
                        fixes += 1
                    break

            # After Set, next non-Block touch with reasonable speed → Attack (kill)
            for j, r in enumerate(rows):
                if r["event"] != "Set":
                    continue
                for k in range(j + 1, len(rows)):
                    nxt = rows[k]
                    if nxt["event"] in ("Rally", "Point"):
                        break
                    if nxt["event"] not in TOUCH_EVENT_NAMES:
                        continue
                    if nxt["event"] == "Block":
                        break
                    if nxt["event"] in ("Dig", "Reception") and float(
                        nxt.get("ball_speed_peak") or 0
                    ) > 90:
                        closed[indices[k]]["event"] = "Attack"
                        closed[indices[k]]["event_conf"] = 0.75
                        fixes += 1
                    break

        if fixes:
            print(f"  [v6 sequence] Applied {fixes} rally-grammar relabel(s)")
        return fixes


def enforce_event_timeline(
    closed: List[dict],
    event_display_sec: float = 0.40,
) -> int:
    """v7: final CSV timeline pass — Point after last touch; compact non-overlapping windows."""
    fixes = 0
    by_rally: Dict[int, List[int]] = defaultdict(list)
    for idx, row in enumerate(closed):
        by_rally[row.get("rally_id", 0)].append(idx)

    for _rid, indices in by_rally.items():
        rows = [closed[i] for i in indices]
        touch_idxs = [
            j for j, r in enumerate(rows) if r["event"] in TOUCH_EVENT_NAMES
        ]
        point_idxs = [j for j, r in enumerate(rows) if r["event"] == "Point"]

        # Compact touch windows around contact start
        for j in touch_idxs:
            start = float(rows[j]["time_start_sec"])
            end = start + event_display_sec
            if abs(float(rows[j]["time_end_sec"]) - end) > 1e-3:
                closed[indices[j]]["time_end_sec"] = round(end, 3)
                closed[indices[j]]["timestamp_end"] = format_timestamp_ms(end)
                fixes += 1
            rows[j] = closed[indices[j]]

        if touch_idxs and point_idxs:
            last_touch = rows[touch_idxs[-1]]
            last_end = float(last_touch["time_end_sec"])
            for j in point_idxs:
                p_start = float(rows[j]["time_start_sec"])
                if p_start < last_end:
                    new_start = last_end
                    new_end = new_start + 0.05
                    closed[indices[j]]["time_start_sec"] = round(new_start, 3)
                    closed[indices[j]]["time_end_sec"] = round(new_end, 3)
                    closed[indices[j]]["timestamp"] = format_timestamp(new_start)
                    closed[indices[j]]["timestamp_start"] = format_timestamp_ms(new_start)
                    closed[indices[j]]["timestamp_end"] = format_timestamp_ms(new_end)
                    fixes += 1
                    rows[j] = closed[indices[j]]

        # Ensure chronological non-decreasing starts within the rally
        prev_end = -1.0
        ordered = sorted(
            range(len(rows)),
            key=lambda j: (float(rows[j]["time_start_sec"]), 0 if rows[j]["event"] != "Point" else 1),
        )
        for j in ordered:
            start = float(closed[indices[j]]["time_start_sec"])
            end = float(closed[indices[j]]["time_end_sec"])
            if start < prev_end - 1e-6 and closed[indices[j]]["event"] == "Point":
                start = prev_end
                end = start + 0.05
                closed[indices[j]]["time_start_sec"] = round(start, 3)
                closed[indices[j]]["time_end_sec"] = round(end, 3)
                closed[indices[j]]["timestamp"] = format_timestamp(start)
                closed[indices[j]]["timestamp_start"] = format_timestamp_ms(start)
                closed[indices[j]]["timestamp_end"] = format_timestamp_ms(end)
                fixes += 1
            prev_end = max(prev_end, float(closed[indices[j]]["time_end_sec"]))

    if fixes:
        print(f"  [v7 timeline] Applied {fixes} event timing correction(s)")
    return fixes


class VolleyballAnalyticsPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        if self.cfg.device == 0 and not torch.cuda.is_available():
            self.cfg.device = "cpu"
        if self.cfg.crops_dir is None:
            self.cfg.crops_dir = self.cfg.project_root / "output" / "jersey_crops"
        self.player_model, self.player_classes = self._load_player_model()
        self.ball_model, self.ball_classes = self._load_ball_model()
        self.jersey = JerseyReader(
            use_gpu=not self.cfg.ocr_on_cpu and torch.cuda.is_available(),
            vote_min=self.cfg.jersey_vote_min,
            crops_dir=self.cfg.crops_dir if self.cfg.save_jersey_crops else None,
            max_jersey_number=self.cfg.max_jersey_number,
            blur_threshold=self.cfg.ocr_blur_threshold,
            brightness_min=self.cfg.ocr_brightness_min,
            brightness_max=self.cfg.ocr_brightness_max,
            min_bbox_area=self.cfg.ocr_min_bbox_area,
            confirm_min_avg_conf=self.cfg.confirm_min_avg_conf,
            ocr_min_vote_conf=self.cfg.ocr_min_vote_conf,
            roster_numbers=self.cfg.roster_numbers,
        )
        self.jersey_regions = JerseyRegionDetector(
            self.cfg.jersey_detector_weights, self.cfg.device, self.cfg.image_size
        )

    def _load_player_model(self) -> Tuple[YOLO, Optional[List[int]]]:
        finetuned = self.cfg.finetuned_player or (
            self.cfg.project_root / "models" / "volleyball_player_best.pt"
        )
        if finetuned.exists() and self.cfg.prefer_finetuned_player:
            return YOLO(str(finetuned)), None
        return YOLO(self.cfg.player_weights), [COCO_PERSON]

    def _load_ball_model(self) -> Tuple[YOLO, Optional[List[int]]]:
        finetuned = self.cfg.finetuned_ball or (
            self.cfg.project_root / "models" / "volleyball_ball_best.pt"
        )
        if finetuned.exists() and self.cfg.prefer_finetuned_ball:
            return YOLO(str(finetuned)), None
        ball_w = self.cfg.ball_weights or self.cfg.player_weights
        return YOLO(ball_w), [COCO_SPORTS_BALL]

    def detect_ball_yolo(
        self,
        frame: np.ndarray,
        court: CourtROI,
        miss_streak: int = 0,
        prefer_xy: Optional[Tuple[float, float]] = None,
        net_y: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """Detect ball with adaptive confidence and trajectory-aware selection (v6)."""
        classes = self.ball_classes
        # Adaptive conf: after miss streak, use fallback threshold inside court ROI
        primary_conf = self.cfg.conf_ball
        if miss_streak >= self.cfg.ball_miss_adaptive_frames:
            primary_conf = self.cfg.conf_ball_fallback

        def predict(conf: float):
            return self.ball_model.predict(
                frame, conf=conf, classes=classes, verbose=False,
                device=self.cfg.device,
                imgsz=max(self.cfg.image_size, self.cfg.ball_image_size),
            )[0]

        r = predict(primary_conf)
        if r.boxes is None or len(r.boxes) == 0:
            r = predict(self.cfg.conf_ball_fallback)
        if r.boxes is None or len(r.boxes) == 0:
            return None

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        candidates: List[Tuple[float, float, float, float]] = []  # cx, cy, conf, score
        for box in r.boxes:
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            if area > frame_area * 0.015:
                continue
            if not court.contains(cx, cy):
                continue
            score = conf
            # Prefer continuation of last trajectory
            if prefer_xy is not None:
                dist = ((cx - prefer_xy[0]) ** 2 + (cy - prefer_xy[1]) ** 2) ** 0.5
                score += max(0.0, 0.35 * (1.0 - dist / max(self.cfg.ball_max_jump_px * 2, 1)))
            # Prefer near-net / rally height band
            if net_y is not None:
                net_dist = abs(cy - net_y) / max(frame_h, 1)
                if net_dist < 0.25:
                    score += 0.08
            candidates.append((cx, cy, conf, score))

        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[3])
        return best[0], best[1], best[2]

    @staticmethod
    def detect_ball_motion(
        frame: np.ndarray, prev_gray: Optional[np.ndarray], court: CourtROI,
    ) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
        motion = None
        if prev_gray is not None:
            diff = cv2.absdiff(gray_b, prev_gray)
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        circles = cv2.HoughCircles(
            gray_b, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=28, minRadius=3, maxRadius=18,
        )
        best = None
        if circles is not None:
            for c in circles[0]:
                cx, cy, _r = int(c[0]), int(c[1]), int(c[2])
                if not court.contains(cx, cy):
                    continue
                score = 0.30
                if motion is not None and 0 <= cy < motion.shape[0] and 0 <= cx < motion.shape[1]:
                    if motion[cy, cx] > 0:
                        score += 0.40
                if best is None or score > best[2]:
                    best = (float(cx), float(cy), score)
        return best, gray_b


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Common Windows installs when PATH is incomplete (e.g. IDE / notebook kernels)
    candidates = [
        Path(r"C:\ffmp\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def finalize_playable_mp4(
    temp_video: Path,
    final_mp4: Path,
    source_video: Optional[Path] = None,
    max_seconds: Optional[float] = None,
) -> Path:
    """Convert MJPEG AVI → H.264 MP4 (mux source audio when available), then delete AVI."""
    ffmpeg = _find_ffmpeg()
    final_mp4 = Path(final_mp4).with_suffix(".mp4")
    temp_video = Path(temp_video)
    if not temp_video.exists() or temp_video.stat().st_size < 1000:
        raise RuntimeError(f"Temp video missing/empty: {temp_video}")

    def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _cleanup_avis() -> None:
        for p in (
            temp_video,
            final_mp4.with_suffix(".avi"),
            final_mp4.parent / f"{final_mp4.stem}.writing.avi",
        ):
            try:
                if p.exists() and p.suffix.lower() == ".avi":
                    p.unlink()
                    print(f"  [v7] Deleted AVI: {p.name}")
            except OSError as exc:
                print(f"  [v7 warning] Could not delete {p.name}: {exc}")

    if not ffmpeg:
        print("  [v7 warning] ffmpeg not found — keeping AVI fallback")
        avi = final_mp4.with_suffix(".avi")
        if temp_video.resolve() != avi.resolve():
            if avi.exists():
                avi.unlink(missing_ok=True)
            shutil.move(str(temp_video), str(avi))
        return avi

    # Write to a temp mp4 first so a failed convert never leaves a half-written client file
    tmp_mp4 = final_mp4.with_suffix(".converting.mp4")
    common_v = [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
    ]

    attempts: List[List[str]] = []
    # 1) Video from analytics AVI + optional audio from source
    if source_video and Path(source_video).exists():
        cmd_audio = [ffmpeg, "-y", "-i", str(temp_video), "-i", str(source_video)]
        cmd_audio.extend(["-map", "0:v:0", "-map", "1:a:0?"])
        cmd_audio.extend(common_v)
        cmd_audio.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
        if max_seconds and max_seconds > 0:
            cmd_audio.extend(["-t", str(max_seconds)])
        cmd_audio.append(str(tmp_mp4))
        attempts.append(cmd_audio)
    # 2) Video-only fallback
    cmd_silent = [ffmpeg, "-y", "-i", str(temp_video)]
    cmd_silent.extend(common_v)
    cmd_silent.append("-an")
    if max_seconds and max_seconds > 0:
        cmd_silent.extend(["-t", str(max_seconds)])
    cmd_silent.append(str(tmp_mp4))
    attempts.append(cmd_silent)

    last_err = ""
    for cmd in attempts:
        if tmp_mp4.exists():
            tmp_mp4.unlink(missing_ok=True)
        r = _run_ffmpeg(cmd)
        if r.returncode == 0 and tmp_mp4.exists() and tmp_mp4.stat().st_size > 1000:
            if final_mp4.exists():
                final_mp4.unlink(missing_ok=True)
            tmp_mp4.replace(final_mp4)
            _cleanup_avis()
            size_mb = final_mp4.stat().st_size / (1024 * 1024)
            has_audio = "1:a:0" in " ".join(cmd)
            print(
                f"  [v7] Converted to H.264 MP4"
                f"{' (with audio)' if has_audio else ' (video only)'}: "
                f"{final_mp4.name} ({size_mb:.1f} MB)"
            )
            return final_mp4
        last_err = ((r.stderr or r.stdout or "")[-400:]).strip()
        print(f"  [v7 warning] ffmpeg attempt failed: {last_err[:180]}")

    print(f"  [v7 warning] All ffmpeg attempts failed — keeping AVI fallback")
    avi = final_mp4.with_suffix(".avi")
    if temp_video.resolve() != avi.resolve():
        if avi.exists():
            avi.unlink(missing_ok=True)
        shutil.move(str(temp_video), str(avi))
    tmp_mp4.unlink(missing_ok=True)
    return avi


# ──────────────────────────────────────────────────────────────────────────────
# Single-CSV output spec (exactly these 11 columns, one file per video)
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV_COLUMNS = [
    "event_id",
    "video_name",
    "rally_id",
    "event_type",
    "time_start_sec",
    "time_end_sec",
    "timestamp",
    "player_track_id",
    "jersey_number",
    "jersey_confidence",
    "event_confidence",
]


def _build_output_row(seg: dict, video_name: str) -> dict:
    """Map an internal closed-segment dict to the clean 11-column output row."""
    return {
        "event_id": seg["event_id"],
        "video_name": video_name,
        "rally_id": seg["rally_id"],
        "event_type": seg["event"],
        "time_start_sec": seg["time_start_sec"],
        "time_end_sec": seg["time_end_sec"],
        "timestamp": seg["timestamp"],
        "player_track_id": seg.get("player_track_id", ""),
        "jersey_number": seg.get("jersey_number", ""),
        "jersey_confidence": seg.get("jersey_conf", ""),
        "event_confidence": seg.get("event_conf", ""),
    }


def cluster_track_identities(
    jersey_cache: Dict[int, str],
    jersey_conf_cache: Dict[int, float],
    track_timeline: Dict[int, dict],
    cfg: "PipelineConfig",
) -> Dict[int, str]:
    """Post-process: merge track IDs that are almost certainly the same physical player.

    Problem this solves (v4):
    When a player goes off-screen and re-appears, YOLO Tracker assigns a NEW track_id.
    So one real player wearing #6 may generate track IDs 3, 24, 61, 180 etc, each
    confirmed independently as #6.  Without linking these, events on track 61 do not
    benefit from jersey confidence accumulated on track 3 (an earlier track).

    This function clusters tracks by:
      1. Same confirmed jersey number
      2. Temporal adjacency: end-time of track A + cluster_max_time_gap_sec >= start of B
      3. Spatial adjacency: last bbox of A and first bbox of B within cluster_max_bbox_gap_px

    Tracks in a cluster with no jersey inherit the cluster representative's jersey.
    Returns the updated jersey_cache (mutated in-place).
    """
    max_time_gap = cfg.cluster_max_time_gap_sec
    max_bbox_gap = cfg.cluster_max_bbox_gap_px

    # Only process tracks that appear in both jersey_cache and the timeline
    confirmed_tids = [tid for tid in jersey_cache if tid in track_timeline]

    # Group confirmed tracks by their jersey number
    by_jersey: Dict[str, List[int]] = defaultdict(list)
    for tid in confirmed_tids:
        by_jersey[jersey_cache[tid]].append(tid)

    # Union-Find helpers
    parent: Dict[int, int] = {tid: tid for tid in confirmed_tids}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Representative = whichever track has higher confidence
        conf_a = jersey_conf_cache.get(ra, 0.0)
        conf_b = jersey_conf_cache.get(rb, 0.0)
        if conf_b > conf_a:
            parent[ra] = rb
        else:
            parent[rb] = ra

    for jersey_num, tids in by_jersey.items():
        if len(tids) < 2:
            continue
        # Sort by first appearance time
        tids_sorted = sorted(tids, key=lambda t: track_timeline[t]["t_start"])
        for i in range(len(tids_sorted) - 1):
            ta = tids_sorted[i]
            tb = tids_sorted[i + 1]
            info_a = track_timeline[ta]
            info_b = track_timeline[tb]
            time_gap = info_b["t_start"] - info_a["t_end"]
            # Must be temporally close (but not overlapping significantly)
            if time_gap < -2.0 or time_gap > max_time_gap:
                continue
            # Spatial check: centroid of last bbox of A vs first bbox of B
            ax = (info_a["last_x1"] + info_a["last_x2"]) / 2.0
            ay = (info_a["last_y1"] + info_a["last_y2"]) / 2.0
            bx = (info_b["first_x1"] + info_b["first_x2"]) / 2.0
            by = (info_b["first_y1"] + info_b["first_y2"]) / 2.0
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist <= max_bbox_gap:
                union(ta, tb)

    # Propagate representative's jersey+confidence to weaker cluster members
    clusters_updated = 0
    for tid in confirmed_tids:
        rep = find(tid)
        if rep != tid and rep in jersey_cache:
            rep_conf = jersey_conf_cache.get(rep, 0.0)
            tid_conf = jersey_conf_cache.get(tid, 0.0)
            # Only override if the representative has strictly higher confidence
            if rep_conf > tid_conf:
                jersey_cache[tid] = jersey_cache[rep]
                jersey_conf_cache[tid] = rep_conf
                clusters_updated += 1

    if clusters_updated:
        print(f"  [cluster] Propagated identity to {clusters_updated} fragmented track(s)")
    return jersey_cache


@torch.inference_mode()
def process_video(
    pipeline: VolleyballAnalyticsPipeline,
    video_path: Path,
    out_path: Path,
    max_seconds: Optional[float] = None,
    split_label: str = "custom",
) -> dict:
    cfg = pipeline.cfg
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seconds * fps) if max_seconds else (total or 10**9)

    # ── v5: Reset jersey state per video (Bug #1 fix) ────────────────────────
    # JerseyReader holds per-track vote accumulators and confirmed identities.
    # If the same pipeline object processes multiple videos sequentially, track
    # IDs from a previous video can bleed into the next one (e.g. tid=3 confirmed
    # as #6 from Video 1 appearing in Video 2's jerseys.json).  Clearing these
    # dicts at video start ensures complete isolation between videos.
    pipeline.jersey._votes.clear()
    pipeline.jersey._vote_counts.clear()
    pipeline.jersey._confirmed.clear()
    pipeline.jersey._last_voted_num.clear()
    pipeline.jersey._consecutive_same.clear()
    print(f"  [v5] Jersey state reset for: {video_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # v5: Output video filename = <video_stem>_analytics.mp4 (no split prefix)
    video_stem = video_path.stem
    out_path = out_path.parent / f"{video_stem}_analytics.mp4"
    temp_path = out_path.parent / f"{video_stem}_analytics.writing.avi"
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    court = CourtROI(w, h, cfg)
    engine = RallyEventEngine(fps=fps, cfg=cfg)
    engine.net_y = h * cfg.net_y_ratio
    ball_tracker = BallKalmanTracker(max_jump_px=cfg.ball_max_jump_px)
    ball_tracker.max_prediction_frames = max(6, int(cfg.ball_predict_frames))
    last_boxes: List[Tuple[int, int, int, int, int, str, float, float]] = []
    jersey_cache: Dict[int, str] = {}
    jersey_conf_cache: Dict[int, float] = {}
    last_ocr_frame: Dict[int, int] = defaultdict(lambda: -(10**9))
    # ── v4: Track timeline ──────────────────────────────────────────────────
    # Records each track's first/last appearance time and bbox for clustering.
    track_timeline: Dict[int, dict] = {}
    current_event = ""
    frame_idx = 0
    t0 = time.time()
    prev_gray: Optional[np.ndarray] = None
    yolo_ball_miss = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps
        player_boxes_evt: List[Tuple[int, int, int, int, int, str, float]] = []

        if frame_idx % cfg.player_infer_stride == 0:
            track_kwargs = dict(
                source=frame, persist=True, conf=cfg.conf_player,
                tracker=cfg.tracker, verbose=False, device=cfg.device, imgsz=cfg.image_size,
            )
            if pipeline.player_classes is not None:
                track_kwargs["classes"] = pipeline.player_classes
            results = pipeline.player_model.track(**track_kwargs)
            r0 = results[0]
            new_boxes = []
            if r0.boxes is not None and len(r0.boxes):
                ids = r0.boxes.id
                for i, box in enumerate(r0.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if (x2 - x1) < 18 or (y2 - y1) < 36:
                        continue
                    tid = int(ids[i].item()) if ids is not None else i
                    det_conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                    jersey = jersey_cache.get(tid, "")
                    jconf = jersey_conf_cache.get(tid, 0.0)

                    # ── v3/v5: Best-frame selection + confirmed-track skip ─────
                    # 1. If already confirmed and skip flag is on, don't waste time on OCR.
                    # v5: Also gate on player height >= ocr_min_player_height_px.
                    run_ocr = False
                    player_height = y2 - y1
                    if cfg.ocr_skip_confirmed and pipeline.jersey.is_confirmed(tid):
                        # Track already locked — just read from cache, no OCR needed.
                        pass
                    elif player_height < cfg.ocr_min_player_height_px:
                        # v5: Player bbox too small for reliable OCR — skip.
                        run_ocr = False
                    else:
                        crop = frame[y1:y2, x1:x2]
                        quality = pipeline.jersey.frame_quality_score(crop)
                        frames_since_ocr = frame_idx - last_ocr_frame[tid]
                        ocr_every_frames = max(1, int(cfg.ocr_interval_sec * fps))
                        ocr_min_frames = max(1, int(cfg.ocr_min_interval_sec * fps))

                        if quality == 0.0:
                            # Frame is definitely bad (too small/blurry/dark) — skip OCR
                            run_ocr = False
                        elif quality >= 0.6 and frames_since_ocr >= ocr_min_frames:
                            # High-quality frame: run OCR more aggressively
                            run_ocr = True
                        elif frames_since_ocr >= ocr_every_frames:
                            # Normal interval elapsed: run OCR if quality is at least marginal
                            run_ocr = quality > 0.0

                        if run_ocr:
                            last_ocr_frame[tid] = frame_idx
                            number_regions = pipeline.jersey_regions.crops(crop)
                            ocr_crop = number_regions[0] if number_regions else crop
                            num, conf = pipeline.jersey.update_track(tid, ocr_crop, frame_idx)
                            if num and conf >= jconf:
                                jersey_cache[tid] = num
                                jersey_conf_cache[tid] = conf
                                jersey, jconf = num, conf
                    # ── end v3/v5 OCR block ───────────────────────────────────

                    player_boxes_evt.append((x1, y1, x2, y2, tid, jersey, jconf))
                    new_boxes.append((x1, y1, x2, y2, tid, jersey, jconf, det_conf))

                    # ── v4: Update track timeline for post-process clustering ──
                    if tid not in track_timeline:
                        track_timeline[tid] = {
                            "t_start": t_sec,
                            "t_end": t_sec,
                            "first_x1": x1, "first_y1": y1,
                            "first_x2": x2, "first_y2": y2,
                            "last_x1": x1, "last_y1": y1,
                            "last_x2": x2, "last_y2": y2,
                        }
                    else:
                        track_timeline[tid]["t_end"] = t_sec
                        track_timeline[tid]["last_x1"] = x1
                        track_timeline[tid]["last_y1"] = y1
                        track_timeline[tid]["last_x2"] = x2
                        track_timeline[tid]["last_y2"] = y2
            last_boxes = new_boxes
        else:
            player_boxes_evt = [(b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in last_boxes]

        ball_det = None
        ball_yolo = None
        ball_accepted = False
        ball_src = "none"
        if engine.consume_ball_reset_flag():
            ball_tracker.reset()
            yolo_ball_miss = 0

        if frame_idx % cfg.infer_stride == 0:
            prefer_xy = None
            if ball_tracker.x is not None and ball_tracker.y is not None:
                prefer_xy = (ball_tracker.x, ball_tracker.y)
            ball_yolo = pipeline.detect_ball_yolo(
                frame,
                court,
                miss_streak=yolo_ball_miss,
                prefer_xy=prefer_xy,
                net_y=engine.net_y,
            )
            if ball_yolo:
                yolo_ball_miss = 0
                ball_accepted = ball_tracker.update(ball_yolo[0], ball_yolo[1], ball_yolo[2], t_sec)
                if ball_accepted and ball_tracker.is_confirmed(cfg.ball_confirm_frames):
                    ball_det = (ball_tracker.x, ball_tracker.y, ball_tracker.conf)
                    ball_src = "yolo"
            else:
                yolo_ball_miss += 1
            if cfg.use_motion_fallback and ball_yolo is None and yolo_ball_miss >= 4:
                motion_ball, prev_gray = pipeline.detect_ball_motion(frame, prev_gray, court)
                if motion_ball is not None:
                    ball_accepted = ball_tracker.update(
                        motion_ball[0], motion_ball[1], motion_ball[2] * 0.75, t_sec
                    )
                    if ball_accepted and ball_tracker.is_confirmed(cfg.ball_confirm_frames):
                        ball_det = (ball_tracker.x, ball_tracker.y, ball_tracker.conf)
                        ball_src = "motion"
            elif prev_gray is None or frame_idx % cfg.infer_stride == 0:
                _, prev_gray = pipeline.detect_ball_motion(frame, prev_gray, court)

        ball = ball_det
        if ball is None and not ball_accepted:
            pred = ball_tracker.predict()
            if pred and pred[2] >= cfg.min_ball_conf_contact:
                ball = pred
                ball_src = "pred"

        emitted = engine.update(frame_idx, ball, player_boxes_evt, h, ball_src=ball_src)
        if emitted:
            current_event = emitted
            if emitted in TOUCH_EVENT_NAMES and engine.closed:
                last_row = engine.closed[-1]
                tid_raw = last_row.get("player_track_id")
                try:
                    contact_tid = int(tid_raw)
                except (TypeError, ValueError):
                    contact_tid = None
                if contact_tid is not None:
                    for (x1, y1, x2, y2, tid, _jersey, _jconf, _dc) in last_boxes:
                        if tid != contact_tid or (y2 - y1) < cfg.ocr_min_player_height_px:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        if pipeline.jersey.frame_quality_score(crop) <= 0.0:
                            break
                        number_regions = pipeline.jersey_regions.crops(crop)
                        ocr_crop = number_regions[0] if number_regions else crop
                        num, conf = pipeline.jersey.update_track(tid, ocr_crop, frame_idx)
                        last_ocr_frame[tid] = frame_idx
                        if num and conf >= cfg.jersey_export_min_conf:
                            jersey_cache[tid] = num
                            jersey_conf_cache[tid] = max(conf, jersey_conf_cache.get(tid, 0.0))
                            last_row["jersey_number"] = num
                            last_row["jersey_conf"] = round(conf, 3)
                        break

        # ── Visualisation ────────────────────────────────────────────────
        vis = frame.copy()
        cv2.rectangle(vis, (court.x1, court.y1), (court.x2, court.y2), (80, 80, 80), 1)
        for (x1, y1, x2, y2, tid, jersey, jconf, _dc) in last_boxes:
            color = (0, 220, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tid}"
            if jersey:
                label += f" #{jersey}({jconf:.2f})"
            cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        if ball is not None:
            bx, by = float(ball[0]), float(ball[1])
            if np.isfinite(bx) and np.isfinite(by) and abs(bx) < 1e5 and abs(by) < 1e5:
                cv2.circle(vis, (int(round(bx)), int(round(by))), 10, (0, 0, 255), 2)
        cv2.line(vis, (0, int(h * cfg.net_y_ratio)), (w, int(h * cfg.net_y_ratio)), (255, 128, 0), 1)
        cv2.rectangle(vis, (0, 0), (w, 44), (0, 0, 0), -1)
        hud = (f"Event: {current_event or '-'} | Players: {len(last_boxes)} | "
               f"Jerseys: {len(jersey_cache)} | {frame_idx}/{max_frames}")
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if cfg.debug_overlay:
            dbg = engine.debug_state
            ball_conf = dbg.get("ball_conf", 0.0)
            pred = ball_tracker.last_predicted
            meas = ball_tracker.last_measured
            debug_lines = [
                f"R{dbg.get('rally_id', 0)} active={dbg.get('rally_active')} touch={dbg.get('touch_index', 0)}",
                f"ball_conf={ball_conf} speed={dbg.get('ball_speed', 0)} miss={yolo_ball_miss} since_ball={dbg.get('sec_since_ball')}",
                f"contacts={dbg.get('contact_candidates', 0)} dead={dbg.get('dead_ball_sec', 0)} lockout={dbg.get('lockout_remaining', 0):.1f}",
                f"last={dbg.get('last_event', '-')}",
            ]
            if meas:
                debug_lines.append(f"measured=({meas[0]:.0f},{meas[1]:.0f})/{meas[2]:.2f}")
            if pred:
                debug_lines.append(f"predicted=({pred[0]:.0f},{pred[1]:.0f})/{pred[2]:.2f}")
            y0 = 68
            cv2.rectangle(vis, (0, 44), (min(w, 520), 44 + 24 * len(debug_lines)), (0, 0, 0), -1)
            for i, line in enumerate(debug_lines):
                cv2.putText(
                    vis,
                    line,
                    (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 240, 255),
                    1,
                )
        writer.write(vis)
        frame_idx += 1
        if torch.cuda.is_available() and frame_idx % 200 == 0:
            torch.cuda.empty_cache()
        if frame_idx % 400 == 0:
            print(f"  frame {frame_idx}/{max_frames} ({time.time()-t0:.0f}s)")

    cap.release()
    writer.release()
    engine.finalize(frame_idx)

    # ── Post-process step 1: backfill jerseys from the per-track OCR cache ──
    # By now OCR has had the entire video to vote; update every event row that
    # was recorded before the jersey was confirmed.
    engine.backfill_jerseys(jersey_cache, jersey_conf_cache)

    # ── Post-process step 2 (v4): cluster fragmented track identities ────────
    # When a player leaves frame and re-appears, the tracker assigns a new
    # track_id.  cluster_track_identities() links consecutive track fragments
    # that share the same jersey number and are spatially/temporally adjacent,
    # propagating the high-confidence identity to the weaker fragment.
    # After clustering, we run backfill again so the newly-linked events are
    # also updated in the closed event list.
    cluster_track_identities(jersey_cache, jersey_conf_cache, track_timeline, cfg)
    engine.backfill_jerseys(jersey_cache, jersey_conf_cache)

    # ── Post-process step 3 (v6): rally sequence normalization ───────────────
    # Apply the volleyball grammar before the v5 cleanup pass.
    _pre_events = {r["event_id"]: r["event"] for r in engine.closed}
    RallySequenceNormalizer(
        serve_to_reception_window_sec=cfg.serve_to_reception_window_sec
    ).apply(engine.closed)

    # ── Post-process step 4 (v5): event logic correctness fixes ──────────────
    # Apply Reception/Dig relabelling, Serve fallback, and Block cleanup rules.
    engine.apply_event_logic_fixes(
        serve_to_reception_window_sec=cfg.serve_to_reception_window_sec
    )

    # ── Post-process step 5 (v7): enforce valid Point / touch timeline ────────
    enforce_event_timeline(engine.closed, event_display_sec=cfg.event_display_sec)

    # v8: rows whose label was changed by the grammar passes are INFERRED, not
    # observed - flag them and cap their confidence so they can be reviewed.
    for _r in engine.closed:
        if _pre_events.get(_r["event_id"]) not in (None, _r["event"]):
            _r["inferred"] = True
            _r["event_conf"] = min(float(_r.get("event_conf") or 0.0), 0.55)

    playable = finalize_playable_mp4(
        temp_path, out_path, source_video=video_path, max_seconds=max_seconds
    )

    # ── Build the single output CSV (11 columns) ──────────────────────────
    video_name = video_path.name
    output_rows = [_build_output_row(seg, video_name) for seg in engine.closed]

    out_csv_dir = cfg.project_root / "output"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    # v5: Use video stem directly as the CSV/JSON name base (no split prefix).
    # output/<video_stem>_events.csv  and  output/<video_stem>_jerseys.json
    video_stem = video_path.stem
    out_csv = out_csv_dir / f"{video_stem}_events.csv"
    pd.DataFrame(output_rows, columns=OUTPUT_CSV_COLUMNS).to_csv(out_csv, index=False)

    # Also save jerseys JSON for reference
    jersey_path = cfg.project_root / "output" / f"{video_stem}_jerseys.json"
    jersey_path.write_text(
        json.dumps({str(k): v for k, v in jersey_cache.items()}, indent=2),
        encoding="utf-8",
    )

    # v8: full-detail debug CSV (reason for every Point, inferred flags, ball
    # frame counts). The client CSV above keeps its 11 columns unchanged.
    pd.DataFrame(engine.closed).to_csv(
        out_csv_dir / f"{video_stem}_events_debug.csv", index=False
    )
    print(f"  [v8] false rallies discarded: {engine.discarded_rallies}")

    return {
        "discarded_rallies": engine.discarded_rallies,
        "video": video_path.name,
        "out_video": str(playable),
        "frames": frame_idx,
        "n_events": len(engine.closed),
        "out_csv": str(out_csv),
        "jerseys": dict(jersey_cache),
        "event_rows": engine.closed,
        "output_rows": output_rows,
    }
