"""Remove temporary upload files that a killed server left behind."""

from __future__ import annotations

import logging
import time

from app import config

logger = logging.getLogger(__name__)

TEMP_UPLOAD_PREFIX = ".incoming-"
TEMP_BACKUP_PREFIX = ".previous-"
STALE_AFTER_SECONDS = 3600


def sweep_stale_uploads(max_age_seconds: int = STALE_AFTER_SECONDS) -> int:
    """Delete temporary upload files older than ``max_age_seconds``; return how many."""
    root = config.ANALYSES_DIR
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for prefix in (TEMP_UPLOAD_PREFIX, TEMP_BACKUP_PREFIX):
        for path in root.glob(f"*/{prefix}*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.warning("Could not remove stale upload file %s", path.name)
    if removed:
        logger.info("Removed %d stale temporary upload file(s)", removed)
    return removed
