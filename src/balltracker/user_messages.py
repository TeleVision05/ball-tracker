"""One place that turns a failure into words a person can act on.

``describe_error`` is the only thing the Streamlit app should show on a
failure. It logs the raw exception (type, message, traceback) and returns a
sentence that says what happened and what to do next. Unknown failures get a
generic "Something went wrong" line; the detail stays in the log.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import (
    CalibrationFailedError,
    CalibrationInputError,
    CalibrationMissingError,
    NoBallFoundError,
    VideoOpenError,
)

logger = logging.getLogger("balltracker.app")

GENERIC_MESSAGE = "Something went wrong. Try again. If it keeps happening, restart the app."

NO_BALL_FOUND = (
    "We couldn't find enough ball positions to work out this delivery. "
    "Pick a clip that starts just before release and ends just after the bounce, "
    "then try again. If you are marking the ball by hand, mark it on at least 5 frames."
)
CALIBRATION_INPUT = (
    "The stump clicks don't line up well enough to set up the camera. "
    "Click the tip and base of each stump again, as close to the stumps as you can."
)
CALIBRATION_FAILED = (
    "We couldn't work out where the camera is from those clicks. "
    "Reset the stump clicks and place them again on a clear, steady frame."
)
CALIBRATION_MISSING = (
    "This video has no camera setup yet. Go back to step 1 and click the stumps first."
)
VIDEO_UNREADABLE = (
    "We couldn't read that video. Check the file still exists and plays in another app, "
    "or re-export it as an MP4, then try again."
)
FILE_MISSING = (
    "A file this step needs could not be found. "
    "Check that the video and camera setup are still where you left them, then try again."
)

_BY_TYPE: tuple[tuple[type[BaseException], str], ...] = (
    (NoBallFoundError, NO_BALL_FOUND),
    (CalibrationInputError, CALIBRATION_INPUT),
    (CalibrationFailedError, CALIBRATION_FAILED),
    (CalibrationMissingError, CALIBRATION_MISSING),
    (VideoOpenError, VIDEO_UNREADABLE),
    (FileNotFoundError, FILE_MISSING),
)


def friendly_message(exc: BaseException) -> str:
    """Plain-language message for ``exc``; never includes the exception text."""
    for exc_type, message in _BY_TYPE:
        if isinstance(exc, exc_type):
            return message
    return GENERIC_MESSAGE


def describe_error(exc: BaseException, context: str = "") -> str:
    """Log the raw failure, then return the message to show the user."""
    logger.error(
        "Failure while %s: %s: %s",
        context or "running",
        type(exc).__name__,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return friendly_message(exc)


CLI_LOG_PATH = Path("out/balltracker-errors.log")


def describe_cli_error(exc: BaseException, context: str = "") -> str:
    """Like ``describe_error`` for command-line scripts.

    The raw detail goes to a log file instead of the terminal, and the returned
    text says where that file is.
    """
    handler: logging.Handler | None = None
    previous_propagate = logger.propagate
    try:
        CLI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(CLI_LOG_PATH, encoding="utf-8")
        logger.addHandler(handler)
        logger.propagate = False
    except OSError:
        handler = None
    try:
        message = describe_error(exc, context)
    finally:
        logger.propagate = previous_propagate
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
    if handler is None:
        return message
    return f"{message} Technical details were saved to {CLI_LOG_PATH}."
