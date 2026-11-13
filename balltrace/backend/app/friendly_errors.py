"""The one place that turns a backend failure into words a person can act on.

Every message says what happened and what to do next. Raw exception text,
status codes and validation-library output are logged here and never returned.
``describe`` is the single entry point used by routes and the analysis pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.errors import (
    AppError,
    OllamaError,
    OllamaModelMissingError,
    OllamaParseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.export_service import FFmpegNotAvailableError
from app.services.video import VideoError

logger = logging.getLogger(__name__)

GENERIC_MESSAGE = "Something went wrong. Try again. If it keeps happening, restart BallTrace."


@dataclass(frozen=True)
class FriendlyError:
    code: str
    status: int
    message: str

    def to_app_error(self) -> AppError:
        return AppError(self.code, self.status, self.message)


# code -> (HTTP status, message template). Templates may use {model} and {limit_mb}.
_CATALOGUE = {
    "upload_too_large": (
        413,
        "That video is larger than the {limit_mb} MB limit. "
        "Trim or compress it, then upload it again.",
    ),
    "video_unreadable": (
        400,
        "We couldn't read that video. Use an MP4 or MOV file that plays on your "
        "computer, then upload it again.",
    ),
    "ffmpeg_unavailable": (
        503,
        "Exporting the video needs FFmpeg, and it isn't working on this computer. "
        "Install FFmpeg and try again. You can still export the detections as JSON or CSV.",
    ),
    "ollama_unavailable": (
        503,
        "BallTrace can't reach Ollama. Start Ollama, check its address in Settings, "
        "then try again.",
    ),
    "ollama_timeout": (
        503,
        "Ollama took too long to answer. Try a smaller model or a larger frame "
        "stride in Settings, then try again.",
    ),
    "ollama_model_missing": (
        503,
        'The model "{model}" isn\'t installed in Ollama. '
        "Install it with `ollama pull {model}` or choose another model in Settings.",
    ),
    "ollama_bad_response": (
        502,
        "The vision model sent back an answer BallTrace couldn't read. "
        "Try again, or choose a different model in Settings.",
    ),
    "analysis_not_found": (
        404,
        "We couldn't find that analysis. It may have been deleted. Refresh the list and try again.",
    ),
    "video_missing": (
        404,
        "This analysis has no video yet, or the file was removed. Upload the video again.",
    ),
    "no_video": (400, "Upload a video to this analysis first, then start it."),
    "export_missing": (404, "There is no exported video yet. Export it first, then download."),
    "invalid_request": (
        422,
        "Some of the information sent wasn't valid. Check the form and try again.",
    ),
    "not_found": (404, "That page or item wasn't found. Refresh the page and try again."),
    "internal_error": (500, GENERIC_MESSAGE),
}


def friendly(code: str, *, model: str = "", limit_mb: int = 0) -> FriendlyError:
    status, template = _CATALOGUE[code]
    return FriendlyError(code, status, template.format(model=model, limit_mb=limit_mb))


def app_error(code: str, **fmt: object) -> AppError:
    """Build a ready-to-raise ``AppError`` for a catalogue code."""
    return friendly(code, **fmt).to_app_error()


def classify(exc: BaseException) -> FriendlyError:
    """Map an exception to its friendly form without logging."""
    if isinstance(exc, OllamaModelMissingError):
        return friendly("ollama_model_missing", model=exc.model)
    if isinstance(exc, OllamaTimeoutError):
        return friendly("ollama_timeout")
    if isinstance(exc, OllamaUnavailableError):
        return friendly("ollama_unavailable")
    if isinstance(exc, OllamaParseError):
        return friendly("ollama_bad_response")
    if isinstance(exc, OllamaError):
        return friendly("ollama_unavailable")
    if isinstance(exc, FFmpegNotAvailableError):
        return friendly("ffmpeg_unavailable")
    if isinstance(exc, VideoError):
        return friendly("video_unreadable")
    return friendly("internal_error")


def describe(exc: BaseException, context: str = "", *, log: bool = True) -> FriendlyError:
    """Log the raw failure (with traceback) and return the message for the user."""
    if log:
        logger.error(
            "Failure while %s: %s: %s",
            context or "handling a request",
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    return classify(exc)
