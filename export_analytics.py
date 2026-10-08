# -*- coding: utf-8 -*-
"""
VolleyScout analytics export (Event Classification & Player Association).

Usage (from project root):
  python export_analytics.py --max-seconds 90 --limit-videos 1   # smoke
  python export_analytics.py --full                                # full match
  python export_analytics.py --video /path/to/match.mp4
  python export_analytics.py --full --video-dir holdout/          # custom folder

Output per video (v7: clean video stem names, no split/smoke/holdout prefix)
-------------------------------------------------------------------------------
  output/<video_stem>_analytics.mp4   annotated video (H.264 MP4 + audio when available)
  output/<video_stem>_events.csv      single clean CSV (11 columns)
  output/<video_stem>_jerseys.json    track_id -> jersey_number map

Merged CSV (output/all_events.csv) is written only with --write-merged-csv
or when processing more than one video.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import pandas as pd
import torch

from volleyball_analytics.pipeline import (
    OUTPUT_CSV_COLUMNS,
    PipelineConfig,
    VolleyballAnalyticsPipeline,
    process_video,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "train"
TEST_DIR = PROJECT_ROOT / "holdout"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4"}


def list_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS)


def readable_videos(paths):
    out = []
    for v in paths:
        cap = cv2.VideoCapture(str(v))
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        cap.release()
        if ok:
            out.append(v)
        else:
            print("SKIP unreadable:", v.name)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="VolleyScout - run event detection & jersey OCR on match videos"
    )
    parser.add_argument(
        "--max-seconds", type=float, default=90.0,
        help="Seconds per video (default 90 smoke test). Use 0 or --full for entire file.",
    )
    parser.add_argument("--full", action="store_true", help="Process entire video(s)")
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Force models/volleyball_player_best.pt",
    )
    parser.add_argument(
        "--pretrained-only", action="store_true",
        help="Ignore fine-tuned weights; use yolo11n COCO person only",
    )
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--video", type=str, default="", help="Process a single video path")
    parser.add_argument(
        "--video-dir", type=str, default="",
        help="Process all videos in this folder (overrides holdout/ auto-discovery)",
    )
    parser.add_argument("--player-weights", type=str, default="yolo11n.pt")
    parser.add_argument("--ball-weights", type=str, default="")
    parser.add_argument("--split", type=str, default="holdout", help="Split label")
    parser.add_argument(
        "--max-jersey-number", type=int, default=25,
        help="Max valid jersey number to filter OCR false positives (default 25)",
    )
    parser.add_argument(
        "--debug-overlay", action="store_true",
        help="Draw ball/tracker/contact/rally debug state on the analytics MP4",
    )
    parser.add_argument(
        "--roster", type=str, default="",
        help="Comma-separated valid jersey numbers for this match, e.g. 1,3,6,12,18",
    )
    parser.add_argument(
        "--jersey-export-min-conf", type=float, default=0.50,
        help="Blank jersey numbers below this confidence in CSV (default 0.50)",
    )
    parser.add_argument(
        "--write-merged-csv", action="store_true",
        help="Also write output/all_events.csv (always on when processing >1 video)",
    )
    args = parser.parse_args()

    max_seconds: float | None = None if (args.full or args.max_seconds <= 0) else args.max_seconds
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    local_player = PROJECT_ROOT / "models" / "volleyball_player_best.pt"
    local_ball   = PROJECT_ROOT / "models" / "volleyball_ball_best.pt"
    run_player   = PROJECT_ROOT / "runs" / "volleyball_player_detection" / "weights" / "best.pt"
    run_ball     = PROJECT_ROOT / "runs" / "volleyball_ball_detection"   / "weights" / "best.pt"

    import shutil
    for src, dst in [(run_player, local_player), (run_ball, local_ball)]:
        if not dst.exists() and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print("Copied training best ->", dst)

    use_finetuned = not args.pretrained_only and (args.finetuned or local_player.exists())
    roster_numbers = None
    if args.roster.strip():
        roster_numbers = {
            int(x.strip())
            for x in args.roster.split(",")
            if x.strip().isdigit()
        }

    cfg = PipelineConfig(
        project_root=PROJECT_ROOT,
        player_weights=args.player_weights,
        ball_weights=args.ball_weights or None,
        prefer_finetuned_player=use_finetuned,
        prefer_finetuned_ball=local_ball.exists(),
        finetuned_player=local_player,
        finetuned_ball=local_ball,
        device=0 if torch.cuda.is_available() else "cpu",
        tracker="botsort.yaml",
        infer_stride=1,
        player_infer_stride=2,
        ocr_interval_sec=3.0,   # 3 s between OCR calls — good balance for long videos
        use_motion_fallback=True,
        ocr_on_cpu=not torch.cuda.is_available(),
        crops_dir=OUTPUT_DIR / "jersey_crops",
        max_jersey_number=args.max_jersey_number,
        roster_numbers=roster_numbers,
        debug_overlay=args.debug_overlay,
        jersey_export_min_conf=args.jersey_export_min_conf,
        ocr_min_bbox_area=5000,
        ocr_blur_threshold=60.0,
        ocr_brightness_min=40.0,
        ocr_brightness_max=230.0,
        ocr_skip_confirmed=True,
        ocr_min_interval_sec=0.5,
        jersey_vote_min=3,
        confirm_min_avg_conf=0.55,
        ocr_min_vote_conf=0.50,
        cluster_max_time_gap_sec=5.0,
        cluster_max_bbox_gap_px=150.0,
        max_rally_sec=25.0,
        ocr_min_player_height_px=100,
        serve_to_reception_window_sec=5.0,
        contact_cooldown_sec=0.55,
        min_ball_speed=25.0,
        contact_dist_px=160.0,
        serve_grace_sec=2.2,
        point_dead_ball_sec=1.4,
        inter_rally_lockout_sec=3.0,
        # v8 - false-rally suppression / ball-aware rally start (tune against ground truth)
        rally_start_min_ball_frames=3,
        min_rally_ball_frames=5,
        min_ball_conf_yolo_contact=0.25,
        jersey_dominance_max_frac=0.35,
    )
    pipeline = VolleyballAnalyticsPipeline(cfg)
    print("Player weights:", local_player if use_finetuned and local_player.exists() else cfg.player_weights)
    print("Ball weights  :", local_ball if local_ball.exists() else (cfg.ball_weights or cfg.player_weights))
    print("Device        :", cfg.device, "| Tracker:", cfg.tracker)
    print(f"min_ball_speed: {cfg.min_ball_speed} px/s  (Set/Attack/Block/Dig gate)")
    print(f"max_jersey_num: {cfg.max_jersey_number} (range 1-{cfg.max_jersey_number})")
    if cfg.roster_numbers:
        print("roster filter :", ",".join(str(n) for n in sorted(cfg.roster_numbers)))
    print(f"debug_overlay : {cfg.debug_overlay}")
    print(f"jersey_min_conf: {cfg.jersey_export_min_conf}")

    # ── Resolve video list ─────────────────────────────────────────────────
    if args.video:
        vpath = Path(args.video).expanduser().resolve()
        if not vpath.exists():
            print("Video not found:", vpath)
            sys.exit(1)
        videos = [(vpath, args.split)]
    elif args.video_dir:
        vdir = Path(args.video_dir).expanduser().resolve()
        videos = [(v, args.split) for v in readable_videos(list_videos(vdir))]
    else:
        test_vids = readable_videos(list_videos(TEST_DIR))
        if not test_vids:
            test_vids = readable_videos(list_videos(TRAIN_DIR))
        if args.limit_videos > 0:
            test_vids = test_vids[: args.limit_videos]
        videos = [(v, args.split) for v in test_vids]

    if not videos:
        print("No readable videos found. Use --video /path/to/file.mp4 or put files in holdout/")
        sys.exit(1)

    print(f"\nProcessing {len(videos)} video(s) | max_seconds={max_seconds}\n")

    all_output_rows: list[dict] = []

    for vpath, split in videos:
        # v5: output filename = <video_stem>_analytics.mp4 (no split/smoke prefix)
        # process_video() writes intermediate .writing.avi, then converts to H.264 MP4 and unlinks the AVI.
        out_vid = OUTPUT_DIR / f"{vpath.stem}_analytics.mp4"
        print(f">>> {vpath.name}\n    -> {out_vid}")

        info = process_video(pipeline, vpath, out_vid, max_seconds=max_seconds, split_label=split)

        all_output_rows.extend(info["output_rows"])

        # Summary
        event_counts = {}
        for row in info["output_rows"]:
            et = row["event_type"]
            event_counts[et] = event_counts.get(et, 0) + 1
        print(f"    events={info['n_events']}  event types={event_counts}")
        print(f"    false rallies discarded={info.get('discarded_rallies', 0)}  (details: output/{vpath.stem}_events_debug.csv)")
        jersey_filled = sum(1 for r in info["output_rows"] if r.get("jersey_number"))
        print(f"    jerseys filled={jersey_filled}/{info['n_events']}")
        print(f"    CSV -> {info['out_csv']}")

    # ── Merged CSV: only when requested or multi-video batch ──────────────
    write_merged = args.write_merged_csv or len(videos) > 1
    print("\n===== DONE =====")
    if write_merged:
        merged_csv = OUTPUT_DIR / "all_events.csv"
        pd.DataFrame(all_output_rows, columns=OUTPUT_CSV_COLUMNS).to_csv(
            merged_csv, index=False
        )
        print(f"Merged events CSV : {merged_csv}  ({len(all_output_rows)} rows)")
    else:
        print("Merged events CSV : skipped (single video; pass --write-merged-csv to enable)")
    print(f"Output folder     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
