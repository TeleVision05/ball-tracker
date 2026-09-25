"""Centralized prompts sent to the Ollama vision model for ball detection.

Keeping these as module-level constants means every caller (initial request,
retry-on-parse-failure, tests) shares the exact same wording. Do not
duplicate this text elsewhere in the codebase.
"""

# Shared JSON schema fragment used by every detection prompt variant.
JSON_SCHEMA_EXAMPLE = """{
"detections": [
{
"frame_index": 0,
"ball_visible": true,
"x": 123,
"y": 456,
"confidence": 0.91
}
]
}"""

DETECTION_PROMPT = f"""You are analyzing consecutive frames from a cricket batting video.

Identify the cricket ball in each frame.

The cricket ball may be very small, blurred, partially occluded, or difficult to see.

Track the same physical cricket ball through the sequence.

Do not guess a ball location.

For every frame, return the center pixel coordinates of the cricket ball if it is visibly identifiable.

If the ball is not visible or you are uncertain, return null.

Return ONLY valid JSON using this exact structure:

{JSON_SCHEMA_EXAMPLE}

Rules:
* x and y are pixel coordinates relative to the image you were given (top-left is 0,0).
* x and y must be the CENTER of the cricket ball.
* confidence must be between 0 and 1.
* Do not invent detections.
* If the ball is not confidently visible, ball_visible must be false and x/y must be null.
* Output the JSON object immediately. Do not write long reasoning."""

SINGLE_FRAME_PROMPT = f"""You are looking at one frame from a cricket batting / net-practice video.

Find the cricket ball. It may be very small, blurred, fast-moving, or partially occluded.

Do not guess. If you cannot confidently see the ball, set ball_visible to false.

Image coordinates: origin (0,0) is the top-left of THIS image. x increases right, y increases down.

Return ONLY valid JSON (no markdown, no prose) using this exact structure:

{JSON_SCHEMA_EXAMPLE}

Rules:
* Use frame_index 0 for this single frame.
* x and y are the CENTER of the cricket ball in this image's pixels.
* confidence is between 0 and 1.
* If not confidently visible: ball_visible false and x/y null."""

CONTACT_SHEET_PROMPT = f"""You are analyzing a horizontal CONTACT SHEET of consecutive cricket video frames.
Each labeled panel is one frame (see frame_index above the panel). Track the same ball across panels.

The cricket ball may be very small, blurred, or partially occluded. Do not guess.

Return ONLY valid JSON using this exact structure:

{JSON_SCHEMA_EXAMPLE}

Rules:
* One detection per panel; frame_index must match the panel label.
* x,y are panel-local (top-left of that panel's video area is 0,0), not full-sheet coordinates.
* x,y are the ball CENTER. confidence in [0,1].
* If uncertain: ball_visible false and x/y null.
* Emit JSON immediately; no long reasoning."""


CORRECTION_PROMPT = f"""Your previous response could not be parsed as valid JSON.

Return ONLY valid JSON using this exact structure, with no markdown formatting, no code fences, and no explanatory text before or after it:

{JSON_SCHEMA_EXAMPLE}

Rules:
* x and y are pixel coordinates relative to the image / panel you were shown (top-left is 0,0).
* x and y must be the CENTER of the cricket ball.
* confidence must be between 0 and 1.
* If the ball is not confidently visible, ball_visible must be false and x/y must be null.
* Return exactly one entry per frame/panel you were shown.
* Output must be a single JSON object and nothing else."""
