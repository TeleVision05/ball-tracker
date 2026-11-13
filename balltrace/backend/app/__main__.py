"""Start the backend: ``python -m app`` (run from ``balltrace/backend``).

Listens on 127.0.0.1 by default. BallTrace has no login, so only change
``BALLTRACE_HOST`` if you understand who can reach that address.
"""

from __future__ import annotations

import ipaddress

import uvicorn

from app import config
from app.logging_config import get_logger, setup_logging


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)
    if not is_loopback(config.HOST):
        logger.warning(
            "BallTrace has no login, and BALLTRACE_HOST=%s makes it reachable from other "
            "computers. Anyone who can reach this address can upload videos and use your "
            "Ollama. Use 127.0.0.1 unless you trust the whole network.",
            config.HOST,
        )
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
