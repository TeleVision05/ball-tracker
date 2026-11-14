"""FastAPI exception handlers: every error body is ``{"detail": ..., "code": ...}``
with ``detail`` already in plain language. Raw exceptions, validation-library
output and status text are logged, never returned."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import AppError
from app.friendly_errors import FriendlyError, describe, friendly

logger = logging.getLogger(__name__)


def _json(error: FriendlyError) -> JSONResponse:
    return JSONResponse({"detail": error.message, "code": error.code}, status_code=error.status)


async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse({"detail": exc.detail, "code": exc.code}, status_code=exc.status)


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Rejected request %s %s: %s", request.method, request.url.path, exc.errors())
    return _json(friendly("invalid_request"))


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(
        "HTTP %s for %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail
    )
    if exc.status_code in (404, 405):
        return _json(friendly("not_found"))
    return _json(friendly("internal_error"))


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    return _json(describe(exc, f"handling {request.method} {request.url.path}"))


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)
