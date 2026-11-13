"""Reject an oversized upload from its Content-Length header, before the body is read.

The route also counts bytes as it copies the file to disk, which covers clients
that send no Content-Length (chunked uploads). This check just stops the common
case early so a huge file is not received at all.
"""

from __future__ import annotations

import re

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app import config
from app.friendly_errors import friendly

_UPLOAD_PATH = re.compile(r"^/api/analyses/[^/]+/upload/?$")

# Multipart framing adds a little to the body beyond the file itself.
_ENVELOPE_ALLOWANCE_BYTES = 1024 * 1024


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._is_upload(scope):
            declared = self._content_length(scope)
            if (
                declared is not None
                and declared > config.max_upload_bytes() + _ENVELOPE_ALLOWANCE_BYTES
            ):
                error = friendly("upload_too_large", limit_mb=config.MAX_UPLOAD_MB)
                response = JSONResponse(
                    {"detail": error.message, "code": error.code},
                    status_code=error.status,
                    headers={"Connection": "close"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _is_upload(scope: Scope) -> bool:
        return scope["method"] == "POST" and bool(_UPLOAD_PATH.match(scope["path"]))

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None
