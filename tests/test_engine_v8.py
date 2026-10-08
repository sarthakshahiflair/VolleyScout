"""Behaviour tests for the v8 rally/point logic (synthetic ball paths).

Run from the project root:  python tests/test_engine_v8.py   (or pytest)
No GPU / weights needed.  torch/ultralytics are stubbed only if not installed.
"""
import contextlib, io, sys, types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
except Exception:  # pragma: no cover
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    t.inference_mode = lambda *a, **k: (lambda f: f)
    sys.modules["torch"] = t
    u = types.ModuleType("ultralytics"); u.YOLO = object; sys.modules["ultralytics"] = u

from volleyball_analytics.pipeline import PipelineConfig, RallyEventEngine  # noqa: E402

FPS = 25.0
PLAYERS = [(150, 480, 250, 640, 1, "", 0.0), (830, 430, 930, 640, 2, "", 0.0),
           (950, 260, 1050, 440, 3, "", 0.0), (700, 250, 800, 430, 4, "", 0.0),
           (60, 500, 160, 660, 5, "", 0.0)]
# serve 2.0 -> reception 3.0 -> set 4.0 -> attack 5.0 -> ball lands 6.2 and stays
RALLY = [(1.5, 200, 560), (2.0, 200, 560), (2.9, 880, 520), (3.0, 880, 520), (3.1, 885, 515),
         (4.0, 1000, 330), (4.1, 1000, 330), (5.0, 750, 320), (5.1, 750, 320),
         (6.2, 300, 690), (9.0, 300, 690)]


def run(keyframes, hidden=(), srcs=(), extra=None, T=14.0):
    cfg = PipelineConfig()
    eng = RallyEventEngine(FPS, cfg); eng.net_y = 720 * 0.42
    ks = np.array(keyframes, float)
    with contextlib.redirect_stdout(io.StringIO()):
        for fi in range(int(T * FPS)):
            t = fi / FPS
            ball, src = None, "yolo"
            if ks[0, 0] <= t <= ks[-1, 0]:
                ball = (float(np.interp(t, ks[:, 0], ks[:, 1])), float(np.interp(t, ks[:, 0], ks[:, 2])), 0.7)
            if any(a <= t < b for a, b in hidden):
                ball = None
            for a, b, name in srcs:
                if a <= t < b:
                    src = name
            if extra and fi in extra:
                ball = extra[fi]
            eng.update(fi, ball, PLAYERS, 720, ball_src=src)
        eng.finalize(int(T * FPS))
    return eng


def events(eng):
    return [(r["event"], r["time_start_sec"]) for r in eng.closed]


def point_t(eng):
    return [t for e, t in events(eng) if e == "Point"]


def test_visible_rally_point_is_backdated():
    eng = run(RALLY)
    assert [e for e, _ in events(eng)][:2] == ["Rally", "Serve"]
    p = point_t(eng)
    assert len(p) == 1 and p[0] <= 6.6, p          # ball stops at 6.2 (old code: ~7.5)


def test_hidden_ball_mid_flight_does_not_end_rally():
    eng = run(RALLY, hidden=[(3.3, 4.9)])
    touches = [e for e, _ in events(eng) if e not in ("Rally", "Point")]
    assert len(touches) >= 5, touches                # old code stopped after 3 touches
    assert point_t(eng)[0] > 6.0


def test_motion_only_ball_cannot_start_rally():
    eng = run([(5.0, 200, 560), (5.1, 260, 520)], srcs=[(5.0, 5.2, "motion")])
    assert events(eng) == []


def test_few_yolo_frames_is_discarded_false_rally():
    ex = {i: (200 + 8 * (i - 100), 560 - 4 * (i - 100), 0.7) for i in range(100, 103)}
    eng = run([(0, 0, 0)], extra=ex)
    assert events(eng) == [] and eng.discarded_rallies == 1


def test_ball_leaving_view_backdates_point_to_last_seen():
    eng = run(RALLY, hidden=[(6.3, 20)])
    assert point_t(eng)[0] <= 6.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
