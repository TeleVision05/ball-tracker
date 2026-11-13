"""Typed failures the user can fix, so the UI can explain them in plain words.

Each class keeps its builtin parent (``RuntimeError`` or ``ValueError``) so
existing ``except RuntimeError`` handlers keep working. The developer-facing
message stays on the exception and goes to the log; ``user_messages`` turns
the type into what a person sees.
"""

from __future__ import annotations


class VideoOpenError(RuntimeError):
    """A video file could not be opened, read or written."""


class NoBallFoundError(RuntimeError):
    """Too few ball positions were found or marked to fit a trajectory."""


class CalibrationMissingError(RuntimeError):
    """No camera setup (stump calibration) exists for this video."""


class CalibrationInputError(ValueError):
    """Stump clicks are missing, too few or do not line up."""


class CalibrationFailedError(RuntimeError):
    """The camera pose could not be solved from the stump clicks."""
