# -*- coding: utf-8 -*-
"""Phase 0 / Phase 5 — smoke, holdout, and rally-level evaluation reports."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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

TOUCH_TYPES = {"Serve", "Reception", "Set", "Attack", "Block", "Dig"}

# Fabio review clip list — priority rallies from the Villadoro client feedback video.
FABIO_REVIEW_RALLIES = [1, 2, 4, 6, 7, 10]


def _event_col(df: pd.DataFrame) -> str:
    if "event_type" in df.columns:
        return "event_type"
    return "event"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def analyze_events_csv(events_csv: Path, duration_sec: float = 90.0) -> Dict[str, Any]:
    df = _safe_read_csv(events_csv)
    if df.empty:
        return {
            "path": str(events_csv), "n_events": 0, "events_per_10s": 0.0,
            "event_counts": {}, "n_rallies": 0, "jerseys_filled": 0,
            "ball_conf_median": None, "ball_speed_median": None,
            "gap_median_sec": None, "single_frame_events": 0,
            "flags": ["no_confirmed_events"],
        }

    ecol = _event_col(df)
    counts = df[ecol].value_counts().to_dict() if ecol in df.columns else {}
    n_rallies = df["rally_id"].nunique() if "rally_id" in df.columns else 0
    jersey_filled = 0
    if "jersey_number" in df.columns:
        jersey_filled = int(
            df["jersey_number"].astype(str).str.strip().replace("nan", "").ne("").sum()
        )

    ball_conf = pd.to_numeric(df.get("ball_conf", pd.Series(dtype=float)), errors="coerce")
    speeds = pd.to_numeric(df.get("ball_speed_peak", pd.Series(dtype=float)), errors="coerce")

    same_frame = 0
    if "frame_start" in df.columns and "frame_end" in df.columns:
        same_frame = int((df["frame_start"] == df["frame_end"]).sum())

    gaps = []
    if "time_start_sec" in df.columns:
        times = sorted(pd.to_numeric(df["time_start_sec"], errors="coerce").dropna())
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    missing_events = sorted(set(EVENT_NAMES) - set(counts.keys()))
    events_per_10s = len(df) / max(duration_sec, 1) * 10

    flags: List[str] = []
    if events_per_10s > 8:
        flags.append("too_many_events_per_10s")
    if n_rallies <= 1 and duration_sec >= 60:
        flags.append("rally_segmentation_stuck")
    if jersey_filled == 0:
        flags.append("no_jerseys_read")
    if missing_events:
        flags.append(f"missing_event_types:{','.join(missing_events)}")
    if same_frame == len(df) and len(df) > 0:
        flags.append("all_single_frame_events")

    return {
        "path": str(events_csv),
        "n_events": len(df),
        "events_per_10s": round(events_per_10s, 2),
        "event_counts": counts,
        "n_rallies": int(n_rallies),
        "jerseys_filled": jersey_filled,
        "ball_conf_median": round(float(ball_conf.median()), 3) if ball_conf.notna().any() else None,
        "ball_speed_median": round(float(speeds.median()), 1) if speeds.notna().any() else None,
        "gap_median_sec": round(float(sorted(gaps)[len(gaps) // 2]), 2) if gaps else None,
        "single_frame_events": same_frame,
        "flags": flags,
    }


def analyze_rallies(events_csv: Path) -> Dict[str, Any]:
    """Rally-level report (no ground truth) for regression / Fabio review."""
    df = _safe_read_csv(events_csv)
    if df.empty or "rally_id" not in df.columns:
        return {
            "path": str(events_csv),
            "n_rallies": 0,
            "serve_only_rallies": 0,
            "serve_only_pct": 0.0,
            "reception_serve_ratio": 0.0,
            "median_touches_per_rally": 0.0,
            "median_serve_duration_sec": None,
            "point_to_next_rally_gaps": {},
            "jersey_fill_rate_touches": 0.0,
            "late_point_rallies": [],
            "fabio_review": [],
            "rallies": [],
            "summary_flags": ["empty_csv"],
        }

    ecol = _event_col(df)
    df = df.copy()
    df["time_start_sec"] = pd.to_numeric(df["time_start_sec"], errors="coerce")
    df["time_end_sec"] = pd.to_numeric(df["time_end_sec"], errors="coerce")
    df["rally_id"] = pd.to_numeric(df["rally_id"], errors="coerce").astype("Int64")

    rallies: List[Dict[str, Any]] = []
    serve_durs: List[float] = []
    touch_counts: List[int] = []
    serve_only = 0
    late_points: List[Dict[str, Any]] = []
    point_end_by_rally: Dict[int, float] = {}
    rally_start_by_id: Dict[int, float] = {}

    for rid, g in df.groupby("rally_id"):
        if pd.isna(rid):
            continue
        rid_i = int(rid)
        g = g.sort_values("time_start_sec")
        seq = [str(x) for x in g[ecol].tolist()]
        touches = g[g[ecol].isin(TOUCH_TYPES)]
        n_touches = len(touches)
        touch_counts.append(n_touches)

        serve_rows = g[g[ecol] == "Serve"]
        serve_dur = None
        if not serve_rows.empty:
            serve_dur = float(
                serve_rows.iloc[0]["time_end_sec"] - serve_rows.iloc[0]["time_start_sec"]
            )
            serve_durs.append(serve_dur)

        mid = set(touches[ecol].tolist()) - {"Serve"}
        is_serve_only = ("Serve" in seq) and (len(mid) == 0)
        if is_serve_only:
            serve_only += 1

        jersey_filled = 0
        if "jersey_number" in touches.columns and n_touches > 0:
            jersey_filled = int(
                touches["jersey_number"].astype(str).str.strip().replace("nan", "").ne("").sum()
            )

        point_rows = g[g[ecol] == "Point"]
        point_end = float(point_rows.iloc[0]["time_end_sec"]) if not point_rows.empty else None
        if point_end is not None:
            point_end_by_rally[rid_i] = point_end

        rally_start = float(g["time_start_sec"].min())
        rally_start_by_id[rid_i] = rally_start

        gap_after_last_touch = None
        if point_end is not None and n_touches > 0:
            last_touch_end = float(touches.iloc[-1]["time_end_sec"])
            gap_after_last_touch = point_end - last_touch_end
            if gap_after_last_touch > 2.0:
                late_points.append({
                    "rally_id": rid_i,
                    "gap_sec": round(gap_after_last_touch, 2),
                    "sequence": " → ".join(seq),
                })

        flags: List[str] = []
        if is_serve_only:
            flags.append("serve_only")
        if "Serve" in seq and "Reception" not in seq and len(mid) >= 1:
            flags.append("missing_reception")
        if serve_dur is not None and serve_dur > 5.0:
            flags.append("long_serve_segment")
        if gap_after_last_touch is not None and gap_after_last_touch > 2.0:
            flags.append("late_point")
        # Point ending before last touch end (impossible timeline)
        if point_end is not None and n_touches > 0:
            last_touch_end = float(touches.iloc[-1]["time_end_sec"])
            if point_end < last_touch_end - 0.05:
                flags.append("invalid_point_timing")

        # Serve → Attack as first post-serve touch (should usually be Reception)
        touch_seq = [str(x) for x in touches[ecol].tolist()]
        if (
            len(touch_seq) >= 2
            and touch_seq[0] == "Serve"
            and touch_seq[1] == "Attack"
        ):
            flags.append("serve_to_attack")

        rallies.append({
            "rally_id": rid_i,
            "sequence": " → ".join(seq),
            "n_touches": n_touches,
            "serve_duration_sec": round(serve_dur, 3) if serve_dur is not None else None,
            "jersey_fill_rate": round(jersey_filled / n_touches, 3) if n_touches else 0.0,
            "time_start_sec": round(rally_start, 3),
            "time_end_sec": round(float(g["time_end_sec"].max()), 3),
            "flags": flags,
            "fabio_priority": rid_i in FABIO_REVIEW_RALLIES,
        })

    # Point → next Rally gaps
    sorted_rids = sorted(rally_start_by_id.keys())
    gaps_pnr: List[float] = []
    for i in range(len(sorted_rids) - 1):
        a, b = sorted_rids[i], sorted_rids[i + 1]
        if a in point_end_by_rally:
            gaps_pnr.append(rally_start_by_id[b] - point_end_by_rally[a])

    n_serve = int((df[ecol] == "Serve").sum())
    n_recv = int((df[ecol] == "Reception").sum())
    n_rallies = len(rallies)

    touch_df = df[df[ecol].isin(TOUCH_TYPES)]
    jersey_fill_rate = 0.0
    if len(touch_df) and "jersey_number" in touch_df.columns:
        filled = touch_df["jersey_number"].astype(str).str.strip().replace("nan", "").ne("")
        jersey_fill_rate = float(filled.mean())

    invalid_point = sum(1 for r in rallies if "invalid_point_timing" in r["flags"])
    serve_to_attack = sum(1 for r in rallies if "serve_to_attack" in r["flags"])

    summary_flags: List[str] = []
    if n_rallies and serve_only / n_rallies > 0.25:
        summary_flags.append("high_serve_only_rate")
    if n_serve and n_recv / max(n_serve, 1) < 0.4:
        summary_flags.append("low_reception_serve_ratio")
    if gaps_pnr and sum(1 for g in gaps_pnr if g < 1.0) / len(gaps_pnr) > 0.15:
        summary_flags.append("many_fast_restarts")
    if invalid_point:
        summary_flags.append("invalid_point_timing")
    if serve_to_attack / max(n_rallies, 1) > 0.15:
        summary_flags.append("many_serve_to_attack")

    fabio_review = [r for r in rallies if r["fabio_priority"]]

    def _med(vals: List[float]) -> Optional[float]:
        return round(float(statistics.median(vals)), 3) if vals else None

    return {
        "path": str(events_csv),
        "n_rallies": n_rallies,
        "serve_only_rallies": serve_only,
        "serve_only_pct": round(100.0 * serve_only / max(n_rallies, 1), 1),
        "reception_serve_ratio": round(n_recv / max(n_serve, 1), 3),
        "median_touches_per_rally": _med([float(x) for x in touch_counts]) or 0.0,
        "median_serve_duration_sec": _med(serve_durs),
        "invalid_point_timing": invalid_point,
        "serve_to_attack_rallies": serve_to_attack,
        "point_to_next_rally_gaps": {
            "median_sec": _med(gaps_pnr),
            "count_lt_1s": int(sum(1 for g in gaps_pnr if g < 1.0)),
            "count_lt_3s": int(sum(1 for g in gaps_pnr if g < 3.0)),
            "n": len(gaps_pnr),
        },
        "jersey_fill_rate_touches": round(jersey_fill_rate, 3),
        "late_point_rallies": late_points[:30],
        "fabio_review_rallies": FABIO_REVIEW_RALLIES,
        "fabio_review": fabio_review,
        "rallies": rallies,
        "summary_flags": summary_flags,
        "event_counts": df[ecol].value_counts().to_dict(),
    }


def compare_to_ground_truth(
    events_csv: Path,
    ground_truth_csv: Path,
    tolerance_sec: float = 1.5,
) -> Dict[str, Any]:
    pred = _safe_read_csv(events_csv)
    gt = _safe_read_csv(ground_truth_csv)
    if pred.empty or gt.empty:
        return {"error": "pred or ground truth empty"}

    pe_col = _event_col(pred)
    ge_col = _event_col(gt)
    results: Dict[str, Any] = {"by_event": {}, "matched": 0, "missed": 0, "extra": 0}
    pred_times = pred.copy()
    gt_times = gt.copy()
    pred_times["time_start_sec"] = pd.to_numeric(pred_times["time_start_sec"], errors="coerce")
    gt_times["time_start_sec"] = pd.to_numeric(gt_times["time_start_sec"], errors="coerce")

    used_gt = set()
    for _, prow in pred_times.iterrows():
        pe, pt = prow.get(pe_col), prow["time_start_sec"]
        if pd.isna(pt):
            continue
        match = None
        for gi, grow in gt_times.iterrows():
            if gi in used_gt:
                continue
            if grow.get(ge_col) == pe and abs(grow["time_start_sec"] - pt) <= tolerance_sec:
                match = gi
                break
        if match is not None:
            used_gt.add(match)
            results["matched"] += 1
        else:
            results["extra"] += 1

    results["missed"] = len(gt_times) - len(used_gt)
    for ev in EVENT_NAMES:
        gt_n = int((gt_times[ge_col] == ev).sum()) if ge_col in gt_times.columns else 0
        pr_n = int((pred_times[pe_col] == ev).sum()) if pe_col in pred_times.columns else 0
        results["by_event"][ev] = {"ground_truth": gt_n, "predicted": pr_n}
    return results


def write_evaluation_report(
    events_csv: Path,
    out_json: Path,
    duration_sec: float = 90.0,
    ground_truth_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "events_analysis": analyze_events_csv(events_csv, duration_sec=duration_sec),
        "rally_analysis": analyze_rallies(events_csv),
    }
    if ground_truth_csv and ground_truth_csv.exists():
        report["ground_truth"] = compare_to_ground_truth(events_csv, ground_truth_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarize_holdout_runs(events_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(events_dir.glob("*_events.csv")):
        stem = p.stem.replace("_events", "")
        analysis = analyze_events_csv(p, duration_sec=90.0)
        rally = analyze_rallies(p)
        rows.append({
            "stem": stem,
            **{k: v for k, v in analysis.items() if k != "path"},
            "serve_only_pct": rally.get("serve_only_pct"),
            "reception_serve_ratio": rally.get("reception_serve_ratio"),
            "median_touches_per_rally": rally.get("median_touches_per_rally"),
            "median_serve_duration_sec": rally.get("median_serve_duration_sec"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    return df


def _print_rally_summary(report: Dict[str, Any]) -> None:
    print(f"path                  : {report.get('path')}")
    print(f"n_rallies             : {report.get('n_rallies')}")
    print(f"serve_only            : {report.get('serve_only_rallies')} "
          f"({report.get('serve_only_pct')}%)")
    print(f"reception/serve ratio : {report.get('reception_serve_ratio')}")
    print(f"median touches/rally  : {report.get('median_touches_per_rally')}")
    print(f"median serve duration : {report.get('median_serve_duration_sec')} s")
    print(f"jersey fill (touches) : {report.get('jersey_fill_rate_touches')}")
    gaps = report.get("point_to_next_rally_gaps") or {}
    print(f"Point→Rally gap med   : {gaps.get('median_sec')} s "
          f"(<{1}s: {gaps.get('count_lt_1s')}/{gaps.get('n')})")
    print(f"flags                 : {report.get('summary_flags')}")
    print(f"event_counts          : {report.get('event_counts')}")
    print("\nFabio priority rallies:")
    for r in report.get("fabio_review") or []:
        print(f"  rally {r['rally_id']:3d}  touches={r['n_touches']}  "
              f"serve_dur={r['serve_duration_sec']}  flags={r['flags']}")
        print(f"           {r['sequence']}")
    late = report.get("late_point_rallies") or []
    if late:
        print(f"\nLate points (first {min(10, len(late))}):")
        for lp in late[:10]:
            print(f"  rally {lp['rally_id']}: +{lp['gap_sec']}s after last touch")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="VolleyScout rally-level evaluation report (no ground truth required)"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to *_events.csv")
    parser.add_argument(
        "--out-json", type=str, default="",
        help="Optional JSON output path (default: <csv_stem>_rally_report.json)",
    )
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="Optional duration for events_per_10s in summary")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rally = analyze_rallies(csv_path)
    _print_rally_summary(rally)

    out_json = Path(args.out_json) if args.out_json else csv_path.with_name(
        csv_path.stem.replace("_events", "") + "_rally_report.json"
    )
    duration = args.duration_sec if args.duration_sec > 0 else 90.0
    write_evaluation_report(csv_path, out_json, duration_sec=duration)
    print(f"\nWrote report -> {out_json}")


if __name__ == "__main__":
    main()
