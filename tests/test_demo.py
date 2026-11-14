"""``balltracker demo``: runs end to end with no footage and fails in plain language."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from balltracker import cli
from balltracker.demo_scene import VideoWriteError


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    out = tmp_path_factory.mktemp("demo")
    return out, cli.main(["demo", "--out", str(out)])


def test_demo_writes_every_output(demo_run: tuple[Path, int]) -> None:
    out, code = demo_run
    assert code == 0
    for name in (
        "delivery.mp4",
        "overlay.mp4",
        "pitch_map.png",
        "summary.json",
        "analysis.json",
        "session.calib.json",
        "demo_config.yaml",
    ):
        assert (out / name).stat().st_size > 0, name


def test_demo_summary_compares_estimate_with_truth(demo_run: tuple[Path, int]) -> None:
    out, _ = demo_run
    summary = json.loads((out / "summary.json").read_text())
    assert "Synthetic only" in summary["what"]
    truth, estimate = summary["truth"], summary["estimate"]
    assert set(estimate) <= set(truth)
    # The demo runs the real detector and fit, so it is not exact; it must be close.
    assert estimate["release_speed_kmh"] == pytest.approx(truth["release_speed_kmh"], rel=0.15)
    assert estimate["bounce_length_m"] == pytest.approx(truth["bounce_length_m"], abs=1.5)
    assert estimate["bounce_line_m"] == pytest.approx(truth["bounce_line_m"], abs=0.15)
    assert summary["tracking"]["points"] >= 0.6 * summary["clip"]["frames_with_ball"]
    assert summary["calibration"]["rmse_px"] < 5.0


def test_demo_prints_truth_next_to_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Simulator truth" in printed
    for label in ("Release speed", "Bounce length", "Bounce line", "Turn"):
        assert label in printed


def test_missing_codec_gives_a_plain_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_codec(*_args, **_kwargs):
        raise VideoWriteError("mp4v")

    monkeypatch.setattr("balltracker.demo.run_demo", no_codec)
    assert cli.main(["demo", "--out", str(tmp_path)]) == 1
    printed = capsys.readouterr().out
    assert "no MPEG-4 encoder" in printed
    assert "Traceback" not in printed
    assert "VideoWriteError" in (tmp_path / "demo.log").read_text()


def test_unexpected_failure_points_to_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args, **_kwargs):
        raise ZeroDivisionError("internal detail")

    monkeypatch.setattr("balltracker.demo.run_demo", boom)
    assert cli.main(["demo", "--out", str(tmp_path)]) == 1
    printed = capsys.readouterr().out
    assert "demo.log" in printed
    assert "internal detail" not in printed
    assert "ZeroDivisionError" in (tmp_path / "demo.log").read_text()


def test_missing_ffmpeg_is_a_note_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("balltracker.demo_media.ffmpeg_available", lambda: False)
    assert cli.main(["demo", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "ffmpeg was not found" in printed
    assert (tmp_path / "overlay.mp4").stat().st_size > 0


def test_benchmark_command_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "ACCURACY.md"
    assert cli.main(["benchmark", "--n", "3", "--seed", "5", "--out", str(report)]) == 0
    text = report.read_text()
    assert text.startswith("# Accuracy: synthetic deliveries only")
    assert "`balltracker benchmark --n 3 --seed 5`" in text


@pytest.mark.parametrize("command", ["demo", "benchmark"])
def test_negative_seed_gets_a_plain_message_not_a_traceback(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main([command, "--seed", "-5"])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "--seed must be 0 or more" in err
    assert "Traceback" not in err


def test_benchmark_defaults_to_a_report_path_that_is_not_committed() -> None:
    args = cli.build_parser().parse_args(["benchmark"])
    assert args.out == cli.DEFAULT_REPORT_PATH
    assert args.out.parts[0] == "out"
