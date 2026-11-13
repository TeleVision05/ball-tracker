"""Exception types shared across the backend (no imports from the app)."""

from __future__ import annotations


class OllamaError(Exception):
    """Base class for all Ollama-related failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama server cannot be reached or answers with a server error."""


class OllamaTimeoutError(OllamaUnavailableError):
    """Raised when Ollama is reachable but does not answer in time."""


class OllamaModelMissingError(OllamaError):
    """Raised when Ollama is running but the requested model is not installed."""

    def __init__(self, model: str, detail: str = ""):
        super().__init__(detail or f"model {model!r} not found")
        self.model = model


class OllamaParseError(OllamaError):
    """Raised when the model's response could not be understood, even after one
    correction retry, or Ollama's reply was not the expected shape."""


class AppError(Exception):
    """A failure that is already worded for the person who triggered it.

    ``detail`` is shown to the user as-is. Raw exception text never goes in it;
    log that instead (see ``app.friendly_errors``).
    """

    def __init__(self, code: str, status: int, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.status = status
        self.detail = detail
