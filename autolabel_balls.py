# -*- coding: utf-8 -*-
"""
Auto-label the volleyball (trajectory + physics, NOT COCO pseudo-labels).

Usage (project root):
  python autolabel_balls.py --video train/match1.mp4 --start 60 --end 600 --out label_auto
  python autolabel_balls.py --video-dir train/ --max-seconds 900 --out label_auto --cvat
  # optional: also feed low-confidence boxes of the current ball model as candidates
  python autolabel_balls.py --video train/match1.mp4 --ball-weights models/volleyball_ball_best.pt

Outputs in --out:
  images/{train,val}, labels/{train,val}   YOLO dataset (time-block split, no leakage)
  data_ball_auto.yaml                      for model.train(data=...)
  review_manifest.csv                      review rows with highest uncertainty first
  cvat_yolo/                               (with --cvat) zip it and import in CVAT as "YOLO 1.1"

NEXT: review the top of review_manifest.csv in CVAT, fix labels, keep 300-500
hand-labelled frames as a test set that is never used for training.
Requires a static camera.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from volleyball_analytics.ball_autolabel import (
    AutoLabelConfig, autolabel_video, export_dataset, make_yolo_detector,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _dead_ball_ranges(events_csv: Path, margin: float = 1.0):
    """Seconds OUTSIDE every rally window (Rally start .. Point end), +-margin."""
    import pandas as pd
    df = pd.read_csv(events_csv)
    wins = []
    for _rid, g in df.groupby("rally_id"):
        wins.append((float(g.time_start_sec.min()) - margin, float(g.time_end_sec.max()) + margin))
    wins.sort()
    rng, prev = [], 0.0
    for a, b in wins:
        if a > prev:
            rng.append((prev, a))
        prev = max(prev, b)
    rng.append((prev, 1e9))
    return rng


def main() -> None:
    ap = argparse.ArgumentParser(description="Trajectory-based ball auto-labelling")
    ap.add_argument("--video", default="")
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--out", default="label_auto")
    ap.add_argument("--start", type=float, default=0.0, help="start second")
    ap.add_argument("--end", type=float, default=0.0, help="end second (0 = to the end)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="process this many seconds from --start")
    ap.add_argument("--label-stride", type=int, default=2, help="keep every k-th labelled frame")
    ap.add_argument("--max-pos", type=int, default=4000)
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="empty-label frames per positive; needs --neg-from-events (default off)")
    ap.add_argument("--neg-from-events", default="",
                    help="events CSV from export_analytics.py; negatives are drawn only OUTSIDE rallies (+-1s)")
    ap.add_argument("--ball-weights", default="", help="optional weak ball model for extra candidates")
    ap.add_argument("--cvat", action="store_true", help="also write cvat_yolo/ folder")
    ap.add_argument("--ignore", default="",
                    help="normalised rects to mask, 'x0,y0,x1,y1;x0,y0,x1,y1' (scoreboards, overlays)")
    args = ap.parse_args()

    ignore = tuple(
        tuple(float(v) for v in r.split(","))
        for r in args.ignore.split(";") if r.strip()
    )
    cfg = AutoLabelConfig(ignore_rects=ignore)
    detector = make_yolo_detector(args.ball_weights) if args.ball_weights else None

    if args.video:
        videos = [Path(args.video)]
    elif args.video_dir:
        videos = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    else:
        raise SystemExit("pass --video or --video-dir")

    neg_ranges = _dead_ball_ranges(Path(args.neg_from_events)) if args.neg_from_events else None
    if args.neg_ratio > 0 and neg_ranges is None:
        print('NOTE: --neg-ratio ignored without --neg-from-events (random negatives may hide unlabelled balls)')
    out = Path(args.out)
    for v in videos:
        end = args.end if args.end > 0 else (args.start + args.max_seconds if args.max_seconds > 0 else None)
        print(f">>> {v.name}  [{args.start}s -> {end if end else 'end'}]")
        labels, fps, size = autolabel_video(v, cfg, start_sec=args.start, end_sec=end, detector=detector)
        det = sum(1 for b in labels.values() if b.source == "detected")
        print(f"    labelled frames: {len(labels)} (detected {det}, interpolated {len(labels) - det})")
        counts = export_dataset(v, labels, fps, size, out, label_stride=args.label_stride,
                                max_pos=args.max_pos, neg_ratio=args.neg_ratio, neg_ranges=neg_ranges,
                                cvat=args.cvat)
        print(f"    exported: {counts}")
    print(f"\nDone. Review: {out / 'review_manifest.csv'}")


if __name__ == "__main__":
    main()
