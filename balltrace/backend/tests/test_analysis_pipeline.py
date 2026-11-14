"""run_analysis against a fake Ollama: stored detections, progress and failures."""

from __future__ import annotations

import functools

import httpx
import pytest

from app.models import database as db
from app.services import analysis_pipeline, progress
from app.services.ollama_client import OllamaClient
from tests.conftest import (
    VIDEO_FRAMES,
    FakeOllama,
    ball_at,
    chat_reply,
    model_missing_reply,
    run,
    text_reply,
)

ANALYSIS_ID = "analysis-1"


@pytest.fixture
def use_fake(monkeypatch):
    """Route the pipeline's OllamaClient through a ``FakeOllama`` transport."""

    def install(fake: FakeOllama) -> FakeOllama:
        monkeypatch.setattr(
            analysis_pipeline,
            "OllamaClient",
            functools.partial(OllamaClient, transport=fake.transport),
        )
        return fake

    return install


def analyze(frame_stride: int = 3, window_size: int = 4) -> None:
    run(
        analysis_pipeline.run_analysis(
            analysis_id=ANALYSIS_ID,
            base_url="http://ollama.test",
            model="fake-model",
            window_size=window_size,
            window_overlap=0,
            frame_stride=frame_stride,
        )
    )


def stored():
    return run(db.get_analysis(ANALYSIS_ID)), run(db.get_detections(ANALYSIS_ID))


def test_per_frame_success_stores_detections_and_reports_progress(analysis_with_video, use_fake):
    # stride 3 over 10 frames -> frames 0, 3, 6, 9, one request each.
    # The frame is downscaled only if wider than 480px, so coordinates are 1:1 here.
    fake = use_fake(FakeOllama([chat_reply([ball_at(40, 30)])]))
    analyze(frame_stride=3)

    analysis, detections = stored()
    assert analysis["status"] == "completed"
    assert analysis["error"] is None
    assert [d["frame_index"] for d in detections] == [0, 3, 6, 9]
    assert all(d["ball_visible"] and d["x"] == 40 and d["y"] == 30 for d in detections)
    assert detections[1]["timestamp"] == pytest.approx(0.3)
    assert detections[0]["normalized_x"] == pytest.approx(40 / 160)
    assert len(fake.chat_payloads()) == 4

    final = progress.snapshot(ANALYSIS_ID)
    assert final["status"] == "completed"
    assert final["total_windows"] == 4
    assert final["current_window"] == 4
    assert final["detections_count"] == 4


def test_per_frame_sends_one_image_for_the_requested_model(analysis_with_video, use_fake):
    fake = use_fake(FakeOllama([chat_reply([ball_at(1, 1)])]))
    analyze(frame_stride=5)
    payload = fake.chat_payloads()[0]
    assert payload["model"] == "fake-model"
    assert len(payload["messages"][0]["images"]) == 1
    assert payload["stream"] is False


def test_a_frame_without_a_ball_is_stored_as_not_visible(analysis_with_video, use_fake):
    no_ball = {"frame_index": 0, "ball_visible": False, "x": None, "y": None, "confidence": 0.0}
    use_fake(FakeOllama([chat_reply([no_ball])]))
    analyze(frame_stride=9)
    _, detections = stored()
    assert detections and all(not d["ball_visible"] and d["x"] is None for d in detections)


def test_window_mode_reads_contact_sheet_panels(analysis_with_video, use_fake, monkeypatch):
    monkeypatch.setattr(analysis_pipeline, "ANALYZE_FRAMES_INDIVIDUALLY", False)
    items = [ball_at(30, 20, frame_index=i) for i in range(4)]
    fake = use_fake(FakeOllama([chat_reply(items)]))
    analyze(frame_stride=1, window_size=4)

    analysis, detections = stored()
    assert analysis["status"] == "completed"
    # 10 frames, window 4, no overlap -> windows [0,4) [4,8) [8,10).
    assert len(fake.chat_payloads()) == 3
    assert [d["frame_index"] for d in detections][:4] == [0, 1, 2, 3]
    assert progress.snapshot(ANALYSIS_ID)["total_windows"] == 3


def test_cancel_stops_before_the_next_frame_and_keeps_what_was_found(analysis_with_video, use_fake):
    fake = use_fake(FakeOllama([chat_reply([ball_at(5, 5)])]))
    fake.on_request = lambda n: progress.request_cancel(ANALYSIS_ID) if n == 2 else None
    analyze(frame_stride=3)

    analysis, detections = stored()
    assert analysis["status"] == "cancelled"
    assert len(fake.chat_payloads()) == 2
    assert [d["frame_index"] for d in detections] == [0, 3]
    assert progress.snapshot(ANALYSIS_ID)["status"] == "cancelled"


def test_unparseable_answer_is_retried_once_with_the_correction_prompt(
    analysis_with_video, use_fake
):
    fake = use_fake(
        FakeOllama(
            [text_reply("I think the ball is near the stumps."), chat_reply([ball_at(7, 8)])]
        )
    )
    analyze(frame_stride=9)
    payloads = fake.chat_payloads()
    first, second = payloads[0]["messages"][0]["content"], payloads[1]["messages"][0]["content"]
    assert first != second
    _, detections = stored()
    assert detections[0]["x"] == 7


def test_missing_video_frames_are_skipped_without_failing(analysis_with_video, use_fake):
    run(db.update_analysis(ANALYSIS_ID, frame_count=VIDEO_FRAMES + 20))
    use_fake(FakeOllama([chat_reply([ball_at(2, 2)])]))
    analyze(frame_stride=10)
    analysis, detections = stored()
    assert analysis["status"] == "completed"
    assert {d["frame_index"] for d in detections} <= set(range(VIDEO_FRAMES))


# A raw detail that must never reach the user: it appears only in the fake server's replies.
RAW_DETAIL = "secret-internal-detail-4242"
TECHNICAL_WORDS = ("Traceback", "HTTP", "httpx", "Errno", "500", "404", RAW_DETAIL)


def assert_plain_language(text: str) -> None:
    assert text
    for word in TECHNICAL_WORDS:
        assert word not in text, f"{word!r} leaked into: {text}"


def test_missing_model_fails_fast_with_install_instructions(analysis_with_video, use_fake):
    fake = use_fake(FakeOllama([model_missing_reply("fake-model")]))
    analyze(frame_stride=3)

    analysis, detections = stored()
    assert analysis["status"] == "failed"
    assert len(fake.chat_payloads()) == 1, "must stop at the first frame, not try all of them"
    assert detections == []
    assert "fake-model" in analysis["error"]
    assert "ollama pull" in analysis["error"]
    assert_plain_language(analysis["error"])
    final = progress.snapshot(ANALYSIS_ID)
    assert final["status"] == "failed"
    assert final["message"] == analysis["error"] or analysis["error"] in final["log"][-1]
    assert all(RAW_DETAIL not in line for line in final["log"])


def test_server_down_aborts_after_three_frames_in_a_row(analysis_with_video, use_fake):
    fake = use_fake(
        FakeOllama([httpx.ConnectError(f"All connection attempts failed {RAW_DETAIL}")])
    )
    analyze(frame_stride=1)

    analysis, _ = stored()
    assert analysis["status"] == "failed"
    assert len(fake.chat_payloads()) == analysis_pipeline.MAX_CONSECUTIVE_OLLAMA_FAILURES
    assert "Ollama" in analysis["error"] and "Start" in analysis["error"]
    assert_plain_language(analysis["error"])
    for line in progress.snapshot(ANALYSIS_ID)["log"]:
        assert RAW_DETAIL not in line and "httpx" not in line and "Traceback" not in line


def test_a_brief_outage_is_survived(analysis_with_video, use_fake):
    down = httpx.ConnectError("connection refused")
    fake = use_fake(FakeOllama([down, down, chat_reply([ball_at(3, 3)])]))
    analyze(frame_stride=3)
    analysis, detections = stored()
    assert analysis["status"] == "completed"
    assert len(fake.chat_payloads()) == 4
    assert [d["frame_index"] for d in detections if d["ball_visible"]] == [6, 9]


def test_timeouts_skip_the_frame_and_say_so_in_plain_language(analysis_with_video, use_fake):
    use_fake(FakeOllama([httpx.ReadTimeout(f"timed out {RAW_DETAIL}")]))
    analyze(frame_stride=3)
    analysis, detections = stored()
    assert analysis["status"] == "completed" and detections == []
    log = progress.snapshot(ANALYSIS_ID)["log"]
    assert any("took too long" in line for line in log)
    assert all(RAW_DETAIL not in line for line in log)


def test_malformed_answers_skip_the_frame_without_leaking_validation_output(
    analysis_with_video, use_fake
):
    use_fake(FakeOllama([text_reply(f'{{"detections": [{{"frame_index": "x"}}]}} {RAW_DETAIL}')]))
    analyze(frame_stride=9)
    analysis, detections = stored()
    assert analysis["status"] == "completed" and detections == []
    log = progress.snapshot(ANALYSIS_ID)["log"]
    assert any("couldn't read" in line for line in log)
    for line in log:
        assert "validation" not in line.lower() and RAW_DETAIL not in line


def test_unexpected_crash_is_logged_and_reported_generically(
    analysis_with_video, use_fake, monkeypatch, caplog
):
    use_fake(FakeOllama([chat_reply([ball_at(1, 1)])]))

    def explode(*_args, **_kwargs):
        raise RuntimeError(f"database exploded {RAW_DETAIL}")

    monkeypatch.setattr(analysis_pipeline.db, "insert_detections", explode)
    analyze(frame_stride=9)
    analysis, _ = stored()
    assert analysis["status"] == "failed"
    assert (
        analysis["error"]
        == "Something went wrong. Try again. If it keeps happening, restart BallTrace."
    )
    assert RAW_DETAIL in caplog.text
