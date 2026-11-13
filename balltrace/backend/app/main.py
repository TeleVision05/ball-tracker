"""FastAPI application entrypoint: app instance, CORS, and startup lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import CORS_ORIGINS, ensure_directories
from app.error_handlers import register_error_handlers
from app.logging_config import get_logger, setup_logging
from app.models.database import init_db
from app.upload_limit import UploadSizeLimitMiddleware

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    await init_db()
    logger.info("BallTrace backend started")
    yield
    logger.info("BallTrace backend shutting down")


app = FastAPI(title="BallTrace API", version="0.1.0", lifespan=lifespan)

# Added before CORS so CORS is the outermost layer and also covers the 413 reply.
app.add_middleware(UploadSizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

app.include_router(router, prefix="/api")
