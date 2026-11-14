"""HTTP surface: friendly error bodies, the upload size limit and the safe defaults."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import __main__ as launcher
from app import config
from app.api import routes
from app.friendly_errors import GENERIC_MESSAGE
from app.main import app
from app.models import database as db
from app.services import export_service
from app.services.ollama_client import OllamaClient
from tests.conftest import FakeOllama, run

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW = "secret-internal-detail-4242"
MB = 1024 * 1024


@pytest.fixture
def client(isolated_data):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def analysis_id(client) -> str:
    return client.post("/api/analyses", json={"name": "t"}).json()["id"]


def upload(client, analysis_id: str, data: bytes, name: str = "clip.mp4"):
    return client.post(
        f"/api/analyses/{analysis_id}/upload", files={"file": (name, data, "video/mp4")}
    )


def assert_friendly(response, status: int, code: str) -> str:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"detail", "code"}
    assert body["code"] == code
    detail = body["detail"]
    assert isinstance(detail, str) and detail
    for leak in (RAW, "Traceback", "pydantic", "field required", "Field required", "loc", "{"):
        assert leak not in detail, detail
    return detail


# --- upload limit -------------------------------------------------------------


def test_oversized_upload_is_rejected_from_content_length(client, analysis_id, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    response = upload(client, analysis_id, b"x" * (3 * MB))
    detail = assert_friendly(response, 413, "upload_too_large")
    assert "1 MB" in detail and "Trim or compress" in detail
    assert not list((config.ANALYSES_DIR / analysis_id).glob("source.*"))


def test_upload_just_over_the_limit_is_refused_by_the_route_and_cleaned_up(
    client, analysis_id, monkeypatch
):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    # Within the multipart-envelope allowance, so only the byte counter can catch it.
    response = upload(client, analysis_id, b"x" * (MB + 4096))
    assert_friendly(response, 413, "upload_too_large")
    assert not list((config.ANALYSES_DIR / analysis_id).glob("source.*"))
    assert run(db.get_analysis(analysis_id))["status"] == "created"


def test_upload_under_the_limit_is_accepted(client, analysis_id, tiny_video):
    response = upload(client, analysis_id, tiny_video.read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uploaded" and body["frame_count"] == 10


def test_limit_is_reported_and_follows_the_setting(client, monkeypatch):
    assert client.get("/api/limits").json() == {"max_upload_mb": config.DEFAULT_MAX_UPLOAD_MB}
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 25)
    assert client.get("/api/limits").json() == {"max_upload_mb": 25}


def test_not_a_video_is_explained_without_paths_or_library_text(client, analysis_id):
    response = upload(client, analysis_id, b"this is plainly not a video")
    detail = assert_friendly(response, 400, "video_unreadable")
    assert "MP4 or MOV" in detail
    for leak in ("Could not open", "source.mp4", "tmp", "Invalid video file"):
        assert leak not in detail


# --- other friendly errors ----------------------------------------------------------


def test_unknown_analysis_is_a_friendly_404(client):
    detail = assert_friendly(client.get("/api/analyses/nope"), 404, "analysis_not_found")
    assert "Refresh" in detail


def test_starting_without_a_video_says_to_upload_one(client, analysis_id):
    detail = assert_friendly(client.post(f"/api/analyses/{analysis_id}/start"), 400, "no_video")
    assert "Upload a video" in detail


def test_missing_ffmpeg_is_503_with_install_advice_not_500(
    client, analysis_id, tiny_video, monkeypatch
):
    upload(client, analysis_id, tiny_video.read_bytes())

    def no_ffmpeg(**_kwargs):
        raise export_service.FFmpegNotAvailableError(f"ffmpeg failed: {RAW} stderr dump")

    monkeypatch.setattr(export_service, "export_trajectory_video", no_ffmpeg)
    response = client.post(f"/api/analyses/{analysis_id}/export/video", json={})
    detail = assert_friendly(response, 503, "ffmpeg_unavailable")
    assert "Install FFmpeg" in detail and "JSON or CSV" in detail


def test_invalid_request_body_hides_validation_library_output(client):
    response = client.post("/api/analyses", json={"nope": 1})
    detail = assert_friendly(response, 422, "invalid_request")
    assert "Check the form" in detail


def test_unknown_route_is_friendly(client):
    assert_friendly(client.get("/api/does-not-exist"), 404, "not_found")


def test_unexpected_failure_is_generic_and_logged(client, monkeypatch, caplog):
    async def boom():
        raise RuntimeError(f"sqlite exploded {RAW}")

    monkeypatch.setattr(db, "get_settings", boom)
    with caplog.at_level(logging.ERROR):
        response = client.get("/api/health")
    detail = assert_friendly(response, 500, "internal_error")
    assert detail == GENERIC_MESSAGE
    assert RAW in caplog.text


def _use_fake_ollama(monkeypatch, fake: FakeOllama) -> None:
    monkeypatch.setattr(
        routes, "OllamaClient", functools.partial(OllamaClient, transport=fake.transport)
    )


def test_settings_report_ollama_down_in_plain_language(client, monkeypatch):
    _use_fake_ollama(monkeypatch, FakeOllama([httpx.ConnectError(f"[Errno 61] {RAW}")]))
    body = client.get("/api/settings/ollama").json()
    assert body["available"] is False
    assert "can't reach Ollama" in body["error"]
    assert RAW not in body["error"] and "Errno" not in body["error"]


def test_settings_report_missing_model_with_the_fix(client, monkeypatch):
    tags = httpx.Response(200, json={"models": [{"name": "other:1b"}]})
    _use_fake_ollama(monkeypatch, FakeOllama([tags]))
    body = client.get("/api/settings/ollama").json()
    assert body["available"] is False
    assert "ollama pull" in body["error"]


# --- safe defaults ----------------------------------------------------------------


def _config_in_clean_env(**env: str) -> list[str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("BALLTRACE_")}
    clean.update(env)
    code = "from app import config; print(config.HOST, config.PORT, config.MAX_UPLOAD_MB)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env=clean, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def test_binds_to_loopback_with_a_sensible_limit_by_default():
    assert _config_in_clean_env() == ["127.0.0.1", "8000", str(config.DEFAULT_MAX_UPLOAD_MB)]
    assert config.DEFAULT_HOST == "127.0.0.1"


def test_limit_can_be_configured_and_bad_values_fall_back():
    assert _config_in_clean_env(BALLTRACE_MAX_UPLOAD_MB="25")[2] == "25"
    assert _config_in_clean_env(BALLTRACE_MAX_UPLOAD_MB="lots")[2] == "500"
    assert _config_in_clean_env(BALLTRACE_MAX_UPLOAD_MB="0")[2] == "500"


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
    ],
)
def test_is_loopback(host, loopback):
    assert launcher.is_loopback(host) is loopback


def test_launcher_uses_the_configured_host_and_warns_when_exposed(monkeypatch, caplog):
    started = {}
    monkeypatch.setattr(launcher.uvicorn, "run", lambda *a, **kw: started.update(kw))
    monkeypatch.setattr(launcher, "setup_logging", lambda: None)

    launcher.main()
    assert started["host"] == "127.0.0.1" and "no login" not in caplog.text

    monkeypatch.setattr(config, "HOST", "0.0.0.0")
    with caplog.at_level(logging.WARNING):
        launcher.main()
    assert started["host"] == "0.0.0.0"
    assert "no login" in caplog.text


# --- re-upload must never damage the good video --------------------------------------


def _stored_video(analysis_id: str) -> Path:
    return Path(run(db.get_analysis(analysis_id))["video_path"])


def _assert_original_intact(client, analysis_id: str, original: bytes, before: dict) -> None:
    served = client.get(f"/api/analyses/{analysis_id}/video")
    assert served.status_code == 200
    assert served.content == original
    after = run(db.get_analysis(analysis_id))
    for key in ("video_path", "video_width", "video_height", "fps", "frame_count", "status"):
        assert after[key] == before[key], key
    leftovers = [p.name for p in (config.ANALYSES_DIR / analysis_id).iterdir()]
    assert leftovers == [Path(before["video_path"]).name], leftovers


@pytest.fixture
def uploaded(client, analysis_id, tiny_video):
    original = tiny_video.read_bytes()
    assert upload(client, analysis_id, original).status_code == 200
    return original, dict(run(db.get_analysis(analysis_id)))


def test_reupload_between_limit_and_envelope_keeps_the_original(
    client, analysis_id, uploaded, monkeypatch
):
    original, before = uploaded
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    response = upload(client, analysis_id, b"x" * (MB + MB // 2), name="a.mp4")
    assert_friendly(response, 413, "upload_too_large")
    _assert_original_intact(client, analysis_id, original, before)


def test_reupload_far_over_the_limit_keeps_the_original(client, analysis_id, uploaded, monkeypatch):
    original, before = uploaded
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    assert_friendly(
        upload(client, analysis_id, b"x" * (3 * MB)),
        413,
        "upload_too_large",
    )
    _assert_original_intact(client, analysis_id, original, before)


def test_chunked_reupload_over_the_limit_keeps_the_original(
    client, analysis_id, uploaded, monkeypatch
):
    original, before = uploaded
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    boundary = "xBOUNDARYx"

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            'filename="c.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
        ).encode()
        for _ in range(5):
            yield b"x" * MB
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        f"/api/analyses/{analysis_id}/upload",
        content=body(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert_friendly(response, 413, "upload_too_large")
    _assert_original_intact(client, analysis_id, original, before)


def test_reupload_of_a_corrupt_file_keeps_the_original(client, analysis_id, uploaded):
    original, before = uploaded
    response = upload(client, analysis_id, b"j" * 100, name="junk.mp4")
    assert_friendly(response, 400, "video_unreadable")
    _assert_original_intact(client, analysis_id, original, before)


def test_failed_first_upload_leaves_no_file_and_start_is_refused(client, analysis_id):
    assert_friendly(upload(client, analysis_id, b"j" * 100), 400, "video_unreadable")
    folder = config.ANALYSES_DIR / analysis_id
    assert not folder.exists() or not list(folder.iterdir())
    assert_friendly(client.post(f"/api/analyses/{analysis_id}/start"), 400, "no_video")


def test_good_reupload_replaces_the_video_and_leaves_one_file(
    client, analysis_id, uploaded, tiny_video
):
    new = tiny_video.read_bytes() + b"\0" * 16  # still readable, different bytes
    response = upload(client, analysis_id, new)
    assert response.status_code == 200
    assert client.get(f"/api/analyses/{analysis_id}/video").content == new
    assert len(list((config.ANALYSES_DIR / analysis_id).iterdir())) == 1


def test_reupload_with_a_different_extension_does_not_orphan_the_old_file(
    client, analysis_id, uploaded, tiny_video
):
    assert upload(client, analysis_id, tiny_video.read_bytes(), name="clip.mov").status_code == 200
    names = [p.name for p in (config.ANALYSES_DIR / analysis_id).iterdir()]
    assert names == ["source.mov"]
