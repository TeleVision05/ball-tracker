"""Characterisation tests for the Streamlit session app (scripts/run_session_app.py).

The app is driven headlessly with Streamlit's AppTest: no browser, no server.
Each test copies the launcher script into a temporary project root whose
``src`` links to the real package, so the app's default video folder and its
``sessions/`` folder land in the temporary directory, never in the work tree.

Clicks on the picture come from a custom component that AppTest cannot click,
so tests place stump points and ball marks through ``session_state``, which is
exactly where the component's click handler puts them.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from balltracker.calib_io import save_calibration
from balltracker.calibration import project_stumps_from_synthetic_camera
from balltracker.landmarks import LANDMARK_ORDER
from balltracker.pitch import stump_landmarks_3d
from balltracker.synthetic import default_camera
from synthetic_clips import make_clip

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path("scripts") / "run_session_app.py"


@pytest.fixture
def app_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway project root; restores the balltracker modules afterwards.

    The launcher purges ``balltracker*`` from ``sys.modules`` and prepends its
    ``src`` to ``sys.path`` on every run, which would leave other test modules
    holding stale classes. Put both back when the test ends.
    """
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / SCRIPT, root / SCRIPT)
    (root / "src").symlink_to(REPO / "src")
    (root / "data" / "raw" / "real").mkdir(parents=True)
    monkeypatch.chdir(root)
    saved_path = list(sys.path)
    saved = {k: m for k, m in sys.modules.items() if k.split(".")[0] == "balltracker"}
    yield root
    for name in [k for k in sys.modules if k.split(".")[0] == "balltracker"]:
        del sys.modules[name]
    sys.modules.update(saved)
    sys.path[:] = saved_path


def _videos(root: Path) -> Path:
    return root / "data" / "raw" / "real"


def _add_clip(root: Path):
    clip = make_clip(root / "clips", 42)
    shutil.copy(clip.video, _videos(root) / "clip.mp4")
    return clip


def _stump_points() -> dict[str, np.ndarray]:
    cam = default_camera()
    names = list(stump_landmarks_3d())
    uv = cam.project(np.stack(list(stump_landmarks_3d().values())))
    return {k: uv[i] for i, k in enumerate(names) if np.all(np.isfinite(uv[i]))}


def _add_calibration(root: Path) -> None:
    calib = project_stumps_from_synthetic_camera(default_camera())
    save_calibration(
        calib, _videos(root) / "session.calib.json", stump_image_points=_stump_points()
    )


def _run(root: Path, at: AppTest | None = None) -> AppTest:
    if at is None:
        at = AppTest.from_file(str(root / SCRIPT), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


def _click(at: AppTest, label: str) -> AppTest:
    _button(at, label).click()
    return _run(Path.cwd(), at)


def _texts(elements) -> list[str]:
    return [e.value for e in elements]


def _step(at: AppTest) -> str:
    return at.radio[0].value


def test_first_run_with_no_videos(app_root: Path) -> None:
    at = _run(app_root)
    assert _texts(at.title) == ["BallTracker — bowl analysis"]
    assert (
        _texts(at.caption)[0] == "Film behind the stumps, then see how fast and where it bounced."
    )
    assert at.radio[0].options == ["1. Set up camera", "2. Analyze a ball", "3. See result"]
    assert _step(at) == "1. Set up camera"
    assert _texts(at.header) == ["Set up your camera"]
    assert "Add a video first (use the sidebar) to set up your camera." in _texts(at.info)
    assert "0 video(s) available" in _texts(at.sidebar.caption)
    assert "Camera setup needed" in _texts(at.sidebar.info)
    assert at.sidebar.selectbox[0].options == ["(new session)"]
    assert at.sidebar.text_input[0].value == str(_videos(app_root))
    assert (app_root / "sessions").is_dir()


def test_camera_setup_places_points_and_saves(app_root: Path) -> None:
    _add_clip(app_root)
    at = _run(app_root)
    assert "1 video(s) available" in _texts(at.sidebar.caption)
    assert at.selectbox(key="setup_video_name").options == ["clip.mp4"]
    assert at.slider(key="setup_frame_idx").max == 63
    at = _click(at, "Load picture")
    assert f"**0 of {len(LANDMARK_ORDER)} points**" in _texts(at.markdown)
    assert "Click: **Bowler end — left stump base**" in _texts(at.info)
    assert _button(at, "Undo").disabled and _button(at, "Save camera setup").disabled

    points = _stump_points()
    at.session_state["calib_points"] = {k: points[k] for k in LANDMARK_ORDER[:7]}
    at = _run(app_root, at)
    assert f"**7 of {len(LANDMARK_ORDER)} points**" in _texts(at.markdown)
    assert "Click: **Batter end — middle stump base**" in _texts(at.info)
    at = _click(at, "Undo")
    assert list(at.session_state["calib_points"]) == LANDMARK_ORDER[:6]
    assert not _button(at, "Save camera setup").disabled

    at = _click(at, "Save camera setup")
    assert (_videos(app_root) / "session.calib.json").exists()
    assert _step(at) == "2. Analyze a ball"
    assert _texts(at.header) == ["Analyze a ball"]
    assert at.session_state["calib_rmse"] < 2.0


def test_camera_setup_reset_and_all_points(app_root: Path) -> None:
    _add_clip(app_root)
    at = _click(_run(app_root), "Load picture")
    at.session_state["calib_points"] = dict(_stump_points())
    at = _run(app_root, at)
    assert "All points placed — save when ready." in _texts(at.success)
    at = _click(at, "Reset")
    assert at.session_state["calib_points"] == {}
    assert at.session_state["calib_rmse"] is None


def test_analyze_step_marking_navigation(app_root: Path) -> None:
    _add_clip(app_root)
    _add_calibration(app_root)
    at = _run(app_root)
    assert _step(at) == "2. Analyze a ball"
    assert "Camera setup saved" in _texts(at.sidebar.success)
    start, end = at.session_state["window_clip.mp4"]
    assert 0 <= start < end <= 63
    duration = f"{(end - start + 1) / 120:.2f} seconds · {end - start + 1} frames"
    assert duration in _texts(at.main.caption)

    at = _click(at, "Start marking the ball")
    assert f"**0** marked · frame **{start}** of **{start}–{end}**" in at.markdown[0].value
    assert _button(at, "← Prev").disabled and _button(at, "Use prediction").disabled
    assert _button(at, "Calculate speed & metrics").disabled
    assert "Mark at least **5** frames (5 more) to calculate." in _texts(at.caption)
    at = _click(at, "Next →")
    assert at.session_state["mark_frame"] == start + 1
    at = _click(at, "Skip frame")
    assert at.session_state["mark_frame"] == start + 2
    at = _click(at, "← Prev")
    assert at.session_state["mark_frame"] == start + 1
    at = _click(at, "Cancel marking")
    assert not at.session_state["mark_mode"]
    _button(at, "Start marking the ball")


def test_marking_with_prediction_and_undo(app_root: Path) -> None:
    clip = _add_clip(app_root)
    _add_calibration(app_root)
    at = _click(_run(app_root), "Start marking the ball")
    start = at.session_state["mark_start"]
    at.session_state["mark_clicks"] = {
        str(f): [float(u), float(v)]
        for f, (u, v) in ((f, clip.track_2d[f]) for f in (start, start + 1))
    }
    at = _click(_click(at, "Skip frame"), "Skip frame")
    assert at.session_state["mark_frame"] == start + 2
    assert sorted(at.session_state["mark_clicks"]) == [start, start + 1], "keys revived as ints"
    assert "**2** marked" in at.markdown[0].value
    assert not _button(at, "Use prediction").disabled
    at = _click(at, "Use prediction")
    clicks = at.session_state["mark_clicks"]
    assert sorted(clicks) == [start, start + 1, start + 2]
    expected = 2 * clip.track_2d[start + 1] - clip.track_2d[start]
    assert clicks[start + 2] == pytest.approx(list(expected))
    assert at.session_state["mark_frame"] == start + 3
    at = _click(at, "Undo last mark")
    assert sorted(at.session_state["mark_clicks"]) == [start, start + 1]
    assert at.session_state["mark_frame"] == start + 2


def test_trim_buttons_move_the_window(app_root: Path) -> None:
    _add_clip(app_root)
    _add_calibration(app_root)
    at = _run(app_root)
    start, end = at.session_state["window_clip.mp4"]
    at.slider(key="preview_clip.mp4").set_value(start + 2)
    at = _click(_run(app_root, at), "Where the clip starts = here")
    assert at.session_state["window_clip.mp4"] == (start + 2, max(start + 2, end))
    at.slider(key="preview_clip.mp4").set_value(end - 3)
    at = _click(_run(app_root, at), "Where the clip ends = here")
    assert at.session_state["window_clip.mp4"] == (start + 2, end - 3)


def test_marked_clicks_produce_a_result(app_root: Path) -> None:
    clip = _add_clip(app_root)
    _add_calibration(app_root)
    at = _click(_run(app_root), "Start marking the ball")
    start, end = at.session_state["mark_start"], at.session_state["mark_end"]
    at.session_state["mark_clicks"] = {
        f: [float(clip.track_2d[f][0]), float(clip.track_2d[f][1])] for f in range(start, end + 1)
    }
    at = _run(app_root, at)
    at = _click(at, "Calculate speed & metrics")
    assert _step(at) == "3. See result"
    assert _texts(at.header) == ["Result"]
    assert re.fullmatch(r"\d+\.\d km/h", at.metric[0].value)
    assert at.metric[0].label == "Speed"
    assert "**Pitch map**" in _texts(at.markdown) and "**Video replay**" in _texts(at.markdown)
    details = [m.value for m in at.markdown if m.value.startswith("Swing:")]
    assert len(details) == 1 and "Length:" in details[0]
    session_files = list((app_root / "sessions").glob("*/session.json"))
    assert len(session_files) == 1
    record = json.loads(session_files[0].read_text(encoding="utf-8"))["deliveries"][0]
    assert isinstance(record["checks"], list)
    assert _texts(at.warning) == record["checks"]

    at = _click(at, "Analyze another ball")
    assert _step(at) == "2. Analyze a ball"
    assert at.session_state["last_delivery_id"] is None


def test_automatic_analyze_produces_a_result(app_root: Path) -> None:
    _add_clip(app_root)
    _add_calibration(app_root)
    at = _run(app_root)
    method = [s for s in at.selectbox if s.label == "Detection method"][0]
    assert method.options == ["auto", "white", "hsv", "motion"]
    method.set_value("hsv")
    at = _click(_run(app_root, at), "Run automatic analyze")
    assert _step(at) == "3. See result"
    assert at.metric[0].label == "Speed"


def test_result_step_without_a_session(app_root: Path) -> None:
    at = _run(app_root)
    at.radio[0].set_value("3. See result")
    at = _run(app_root, at)
    assert _texts(at.header) == ["Result"]
    assert "Analyze a ball first." in _texts(at.info)
    at = _click(at, "Go to step 2")
    assert _step(at) == "2. Analyze a ball"
    assert "Set up your camera first." in _texts(at.warning)
    at = _click(at, "Go to camera setup")
    assert _step(at) == "1. Set up camera"


def test_loading_an_existing_session(app_root: Path) -> None:
    from balltracker.session import create_session

    session = create_session(root=app_root / "sessions")
    at = _run(app_root)
    picker = at.sidebar.selectbox[0]
    assert picker.options == ["(new session)", session.id]
    picker.set_value(session.id)
    at = _click(_run(app_root, at), "Load")
    assert f"Loaded {session.id}" in _texts(at.sidebar.success)
    at = _run(app_root, at)
    assert f"Active: `{session.id}`" in _texts(at.sidebar.caption)


def test_result_page_warns_about_an_untrustworthy_saved_result(app_root: Path) -> None:
    """A delivery saved before result checks existed still gets its warnings.

    The metrics are those of a real saved session: the fit stopped on its
    slowest allowed release and missed the marked points.
    """
    from balltracker.result_checks import POOR_FIT
    from balltracker.session import create_session

    session = create_session(root=app_root / "sessions")
    session.deliveries.append(
        {
            "delivery_id": "delivery_0001",
            "source_video_path": str(app_root / "gone.mp4"),
            "metrics": {
                "release_speed_kmh": 29.5,
                "pitch_line_m": -0.24,
                "pitch_length_m": 17.79,
                "bounce_y_m": 2.33,
                "confidence": {"trajectory_fit_ok": False},
            },
        }
    )
    session.save()
    at = _run(app_root)
    at.sidebar.selectbox[0].set_value(session.id)
    at = _click(_run(app_root, at), "Load")
    at.radio[0].set_value("3. See result")
    at = _run(app_root, at)
    warnings = _texts(at.warning)
    assert warnings[0] == POOR_FIT.format(detail="")
    assert warnings[1].startswith("A release speed of 30 km/h is outside what real bowling")
