"""OllamaClient against a fake server: success, model missing, server down,
malformed replies and timeouts. No real Ollama is involved."""

from __future__ import annotations

import base64

import httpx
import pytest

from app.errors import (
    OllamaModelMissingError,
    OllamaParseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.friendly_errors import classify
from app.services.ollama_client import OllamaClient
from tests.conftest import (
    FakeOllama,
    ball_at,
    chat_reply,
    model_missing_reply,
    run,
    text_reply,
)

IMAGE_B64 = base64.b64encode(b"not-really-a-jpeg").decode()


def client_for(fake: FakeOllama, model: str = "fake-model") -> OllamaClient:
    return OllamaClient("http://ollama.test/", model, timeout=3.0, transport=fake.transport)


def tags_reply(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in names]})


# --- success -----------------------------------------------------------------


def test_detect_single_frame_returns_the_ball_position():
    fake = FakeOllama([chat_reply([ball_at(120.5, 64, frame_index=7)])])
    item = run(client_for(fake).detect_single_frame(IMAGE_B64))
    assert (item.ball_visible, item.x, item.y, item.confidence) == (True, 120.5, 64, 0.9)
    assert item.frame_index == 0, "single-frame calls always report local frame 0"


def test_request_goes_to_chat_endpoint_with_model_image_and_no_streaming():
    fake = FakeOllama([chat_reply([ball_at(1, 2)])])
    run(client_for(fake).detect_single_frame(IMAGE_B64))
    (request,) = fake.requests
    assert str(request.url) == "http://ollama.test/api/chat"
    payload = fake.chat_payloads()[0]
    assert payload["model"] == "fake-model"
    assert payload["messages"][0]["images"] == [IMAGE_B64]
    assert payload["stream"] is False and payload["think"] is False


def test_json_inside_a_code_fence_is_accepted():
    fake = FakeOllama([chat_reply([ball_at(9, 9)], wrap_in_fence=True)])
    assert run(client_for(fake).detect_single_frame(IMAGE_B64)).x == 9


def test_answer_in_the_thinking_field_is_used_when_content_is_empty():
    body = '{"detections": [{"frame_index": 0, "ball_visible": true, "x": 4, "y": 5}]}'
    reply = httpx.Response(200, json={"message": {"content": "", "thinking": body}})
    assert run(client_for(FakeOllama([reply])).detect_single_frame(IMAGE_B64)).y == 5


def test_empty_detection_list_means_ball_not_visible():
    item = run(client_for(FakeOllama([chat_reply([])])).detect_single_frame(IMAGE_B64))
    assert item.ball_visible is False and item.x is None


# --- model missing -------------------------------------------------------------


def test_unknown_model_raises_model_missing_with_the_model_name():
    fake = FakeOllama([model_missing_reply("fake-model")])
    with pytest.raises(OllamaModelMissingError) as caught:
        run(client_for(fake).detect_single_frame(IMAGE_B64))
    assert caught.value.model == "fake-model"
    assert len(fake.requests) == 1, "a missing model is not retried"


def test_other_404_is_not_mistaken_for_a_missing_model():
    fake = FakeOllama([httpx.Response(404, text="404 page not found")])
    with pytest.raises(OllamaUnavailableError) as caught:
        run(client_for(fake).detect_single_frame(IMAGE_B64))
    assert not isinstance(caught.value, OllamaModelMissingError)


# --- server down -----------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("All connection attempts failed"),
        httpx.RemoteProtocolError("Server disconnected without sending a response"),
        httpx.Response(503, text="service unavailable"),
        httpx.Response(500, json={"error": "llama runner process has terminated"}),
    ],
    ids=["refused", "dropped", "503", "500"],
)
def test_unreachable_or_failing_server_raises_unavailable(failure):
    with pytest.raises(OllamaUnavailableError) as caught:
        run(client_for(FakeOllama([failure])).detect_single_frame(IMAGE_B64))
    assert not isinstance(caught.value, OllamaTimeoutError)


# --- timeout ---------------------------------------------------------------------


def test_slow_server_raises_a_timeout_which_is_also_unavailable():
    fake = FakeOllama([httpx.ReadTimeout("read timed out")])
    with pytest.raises(OllamaTimeoutError) as caught:
        run(client_for(fake).detect_single_frame(IMAGE_B64))
    assert isinstance(caught.value, OllamaUnavailableError)
    assert "3s" in str(caught.value)


# --- malformed responses ---------------------------------------------------------


def test_reply_that_is_not_json_raises_parse_error():
    fake = FakeOllama([httpx.Response(200, text="<html>proxy error</html>")])
    with pytest.raises(OllamaParseError):
        run(client_for(fake).detect_single_frame(IMAGE_B64))


def test_reply_that_is_a_json_list_raises_parse_error():
    with pytest.raises(OllamaParseError):
        run(
            client_for(FakeOllama([httpx.Response(200, json=[1, 2, 3])])).detect_single_frame(
                IMAGE_B64
            )
        )


def test_reply_without_any_text_raises_parse_error():
    reply = httpx.Response(200, json={"message": {"content": "", "thinking": ""}})
    with pytest.raises(OllamaParseError):
        run(client_for(FakeOllama([reply])).detect_single_frame(IMAGE_B64))


def test_garbled_answer_is_retried_once_then_gives_up():
    fake = FakeOllama([text_reply("The ball is somewhere on the pitch.")])
    with pytest.raises(OllamaParseError):
        run(client_for(fake).detect_single_frame(IMAGE_B64))
    first, second = (p["messages"][0]["content"] for p in fake.chat_payloads())
    assert len(fake.requests) == 2 and first != second


def test_schema_violation_is_retried_and_a_good_second_answer_wins():
    bad = text_reply('{"detections": [{"frame_index": "zero"}]}')
    fake = FakeOllama([bad, chat_reply([ball_at(8, 6)])])
    assert run(client_for(fake).detect_single_frame(IMAGE_B64)).x == 8
    assert len(fake.requests) == 2


def test_truncated_json_is_repaired_without_a_retry():
    cut = '{"detections": [{"frame_index": 0, "ball_visible": true, "x": 3, "y": 4, "confidence": 0.5}, {"frame_ind'
    fake = FakeOllama([text_reply(cut)])
    assert run(client_for(fake).detect_single_frame(IMAGE_B64)).x == 3
    assert len(fake.requests) == 1


# --- health check ----------------------------------------------------------------


def test_check_available_ok_when_model_is_installed():
    fake = FakeOllama([tags_reply("other:1b", "fake-model")])
    assert run(client_for(fake).check_available()) == (True, None)
    assert fake.requests[0].url.path == "/api/tags"


def test_check_available_accepts_latest_tag_for_a_bare_model_name():
    fake = FakeOllama([tags_reply("llava:latest")])
    assert run(client_for(fake, model="llava").check_available()) == (True, None)


def test_check_available_reports_missing_model_in_plain_language():
    ok, message = run(client_for(FakeOllama([tags_reply("other:1b")])).check_available())
    assert ok is False
    assert "fake-model" in message and "ollama pull fake-model" in message


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx.ConnectError("[Errno 61] Connection refused"), "can't reach Ollama"),
        (httpx.ReadTimeout("read timed out"), "too long"),
        (httpx.Response(500, text="boom"), "can't reach Ollama"),
    ],
    ids=["down", "timeout", "500"],
)
def test_check_available_never_returns_raw_error_text(failure, expected):
    ok, message = run(client_for(FakeOllama([failure])).check_available())
    assert ok is False and expected in message
    for raw in ("Errno", "Connection refused", "500", "boom", "httpx"):
        assert raw not in message


def test_check_available_does_not_block_on_an_unreadable_tag_list():
    fake = FakeOllama([httpx.Response(200, text="not json")])
    assert run(client_for(fake).check_available()) == (True, None)


# --- friendly mapping of the client's errors ---------------------------------------


@pytest.mark.parametrize(
    ("exc", "code", "status"),
    [
        (OllamaModelMissingError("m"), "ollama_model_missing", 503),
        (OllamaTimeoutError("x"), "ollama_timeout", 503),
        (OllamaUnavailableError("x"), "ollama_unavailable", 503),
        (OllamaParseError("x"), "ollama_bad_response", 502),
        (RuntimeError("anything else"), "internal_error", 500),
    ],
)
def test_client_errors_map_to_friendly_codes(exc, code, status):
    friendly = classify(exc)
    assert (friendly.code, friendly.status) == (code, status)
