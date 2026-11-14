"""Async HTTP client for Ollama's /api/chat vision endpoint, plus robust
parsing of the JSON detection payload the model is asked to return.

Ollama (via qwen3-vl or any other local vision model) is the ONLY ball
detector in this system. This module never inspects pixel values itself --
it only talks to the model and validates/repairs the JSON it sends back.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import ValidationError

from app.config import (
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_HEALTH_TIMEOUT,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
)
from app.errors import (
    OllamaError,
    OllamaModelMissingError,
    OllamaParseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.friendly_errors import describe, friendly
from app.models.schemas import RawDetectionItem, RawDetectionResponse
from app.prompts.ball_detection import (
    CONTACT_SHEET_PROMPT,
    CORRECTION_PROMPT,
    SINGLE_FRAME_PROMPT,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaModelMissingError",
    "OllamaParseError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
]

logger = logging.getLogger(__name__)

_CODE_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"```\s*$")


# ---------------------------------------------------------------------------
# Pure JSON extraction / repair helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = _CODE_FENCE_OPEN_RE.sub("", text)
    text = _CODE_FENCE_CLOSE_RE.sub("", text)
    return text.strip()


def extract_json_candidate(text: str) -> str:
    """Strip markdown fences and return the substring starting at the first
    '{'. Raises ValueError if no '{' is present at all."""
    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    return cleaned[start:]


def _repair_truncated_json(candidate: str) -> Optional[str]:
    """Attempt to recover a valid JSON object from text that was truncated
    mid-stream (e.g. the model's output was cut off).

    Strategy: scan the string tracking bracket depth (respecting string
    literals), remembering the position right after the most recent fully
    *closed* bracket (object or array) along with the bracket stack at that
    point. Only closing-bracket positions are treated as safe -- a trailing
    comma followed by a half-written element is discarded entirely, since
    keeping it would produce a syntactically-complete-but-semantically-empty
    fragment (e.g. `{"frame_index": 2}` missing required fields). If the
    string ends with unclosed brackets, truncate back to that last safe
    point and append the closing brackets needed to balance it.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    last_safe_idx: Optional[int] = None
    last_safe_stack: Optional[List[str]] = None
    closers = {"{": "}", "[": "]"}

    for i, ch in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe_idx = i + 1
            last_safe_stack = list(stack)

    if not stack:
        # Balanced already; whatever failed to parse wasn't a truncation issue.
        return None
    if last_safe_idx is None or last_safe_stack is None:
        return None

    truncated = candidate[:last_safe_idx].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1]

    suffix = "".join(closers[b] for b in reversed(last_safe_stack))
    return truncated + suffix


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from raw model output, tolerating
    markdown fences, trailing prose, and mid-stream truncation."""
    candidate = extract_json_candidate(text)
    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(candidate)
        return obj
    except json.JSONDecodeError:
        pass

    repaired = _repair_truncated_json(candidate)
    if repaired is not None:
        try:
            obj, _ = decoder.raw_decode(repaired)
            return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {candidate[:200]!r}")


def parse_detection_response(text: str) -> RawDetectionResponse:
    """Parse + validate a model response into a RawDetectionResponse.

    Raises ValueError on any failure (missing JSON, malformed JSON that
    cannot be repaired, or JSON that doesn't match the expected schema).
    """
    try:
        obj = parse_json_object(text)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    try:
        return RawDetectionResponse.model_validate(obj)
    except ValidationError as exc:
        raise ValueError(f"Detection JSON failed schema validation: {exc}") from exc


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------


def _is_model_not_found(response: httpx.Response) -> bool:
    """Ollama answers 404 with {"error": "model ... not found"} for an unknown model."""
    if response.status_code != 404:
        return False
    text = response.text.lower()
    return "model" in text and "not found" in text


def _model_missing(tags: Any, model: str) -> bool:
    """True when /api/tags lists models and ``model`` is not among them."""
    models = tags.get("models") if isinstance(tags, dict) else None
    if not isinstance(models, list):
        return False
    installed = set()
    for entry in models:
        if isinstance(entry, dict):
            installed.update(str(entry.get(k)) for k in ("name", "model") if entry.get(k))
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    return not (wanted & installed)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = OLLAMA_CHAT_TIMEOUT,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Tests inject an ``httpx.MockTransport`` here instead of a real server.
        self._transport = transport

    def _http(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def check_available(self) -> Tuple[bool, Optional[str]]:
        """Confirm the Ollama server is reachable and has the model installed.

        The second value is a plain-language message for the user (never raw
        exception text); the raw failure is logged.
        """
        try:
            async with self._http(OLLAMA_HEALTH_TIMEOUT) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                tags = resp.json()
        except httpx.TimeoutException as exc:
            return False, describe(OllamaTimeoutError(str(exc)), "checking Ollama").message
        except httpx.HTTPError as exc:
            return False, describe(OllamaUnavailableError(str(exc)), "checking Ollama").message
        except ValueError:
            # Reachable but not a tags list: do not block the user on a guess.
            logger.warning("Ollama /api/tags returned a body that is not JSON")
            return True, None

        if _model_missing(tags, self.model):
            logger.warning("Ollama is running but model %r is not installed", self.model)
            return False, friendly("ollama_model_missing", model=self.model).message
        return True, None

    async def _post_chat(self, payload: Dict[str, Any], image_count: int) -> Dict[str, Any]:
        """POST to /api/chat and return the decoded JSON; map failures to typed errors."""
        try:
            async with self._http(self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.timeout:.0f}s "
                f"(model={self.model}, images={image_count})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if _is_model_not_found(exc.response):
                raise OllamaModelMissingError(self.model, exc.response.text[:200]) from exc
            raise OllamaUnavailableError(
                f"Ollama answered with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise OllamaUnavailableError(f"Ollama request failed: {detail}") from exc
        except ValueError as exc:
            raise OllamaParseError(f"Ollama reply was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise OllamaParseError(f"Ollama reply was {type(data).__name__}, expected an object")
        return data

    async def _chat(self, prompt: str, images_b64: List[str]) -> str:
        # qwen3-vl (and other "thinking" VLMs) may emit long chain-of-thought
        # into `message.thinking` and leave `content` empty when the token
        # budget runs out. We:
        #  - disable thinking when the server supports it (`think: false`)
        #  - reserve a generous num_predict so JSON can still be emitted
        #  - fall back to parsing `thinking` if content is blank
        # Do NOT set `format: "json"` for these models — it often traps the
        # JSON inside `thinking` and leaves `content` empty.
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "temperature": OLLAMA_TEMPERATURE,
            },
        }
        data = await self._post_chat(payload, len(images_b64))

        message = data.get("message")
        if not isinstance(message, dict):
            message = {}
        content = str(message.get("content") or "").strip()
        thinking = str(message.get("thinking") or "").strip()
        combined = content or thinking
        if not combined:
            raise OllamaParseError(
                "Ollama returned empty content and thinking "
                f"(done_reason={data.get('done_reason')!r})"
            )
        if not content and thinking:
            logger.warning(
                "Ollama content empty; falling back to thinking field "
                "(done_reason=%r, thinking_chars=%d)",
                data.get("done_reason"),
                len(thinking),
            )
        return combined

    async def _detect_with_retry(
        self, prompt: str, images_b64: List[str], correction_extra: str = ""
    ) -> List[RawDetectionItem]:
        content = await self._chat(prompt, images_b64)
        try:
            parsed = parse_detection_response(content)
            return parsed.detections
        except ValueError as first_error:
            logger.warning(
                "Initial detection JSON parse failed, retrying with correction prompt: %s",
                first_error,
            )
            correction = CORRECTION_PROMPT
            if correction_extra:
                correction = f"{CORRECTION_PROMPT}\n\n{correction_extra}"
            content2 = await self._chat(correction, images_b64)
            try:
                parsed2 = parse_detection_response(content2)
                return parsed2.detections
            except ValueError as second_error:
                raise OllamaParseError(
                    f"Failed to parse detection JSON after correction retry: {second_error}"
                ) from second_error

    async def detect_single_frame(self, image_b64: str) -> RawDetectionItem:
        """Ask the vision model to locate the ball in one frame."""
        items = await self._detect_with_retry(SINGLE_FRAME_PROMPT, [image_b64])
        if not items:
            return RawDetectionItem(
                frame_index=0, ball_visible=False, x=None, y=None, confidence=0.0
            )
        item = items[0]
        # Normalize local index — callers treat this as the only frame.
        return RawDetectionItem(
            frame_index=0,
            ball_visible=item.ball_visible,
            x=item.x,
            y=item.y,
            confidence=item.confidence,
        )

    async def detect_balls_from_contact_sheet(
        self, contact_sheet_b64: str, frame_count: int
    ) -> List[RawDetectionItem]:
        """Send a labeled contact-sheet image covering frame_index 0..N-1."""
        prompt = (
            f"{CONTACT_SHEET_PROMPT}\n\n"
            f"This contact sheet contains exactly {frame_count} panels "
            f"(frame_index 0..{frame_count - 1}). Emit the JSON object now."
        )
        return await self._detect_with_retry(
            prompt,
            [contact_sheet_b64],
            correction_extra=(
                f"There are exactly {frame_count} panels (frame_index 0..{frame_count - 1})."
            ),
        )
