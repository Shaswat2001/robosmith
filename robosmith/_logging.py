"""Logging compatibility layer.

RoboSmith uses Loguru when it is installed, but imports should still work in
minimal environments where optional tests or metadata-only commands do not need
the full runtime dependency set.
"""

from __future__ import annotations

import logging
from typing import Any


try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger("robosmith")

    class _FallbackLogger:
        def add(self, *args: Any, **kwargs: Any) -> int:
            return 0

        def remove(self, *args: Any, **kwargs: Any) -> None:
            return None

        def bind(self, **kwargs: Any) -> "_FallbackLogger":
            return self

        def opt(self, **kwargs: Any) -> "_FallbackLogger":
            return self

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.debug(message, *args, **kwargs)

        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.info(message, *args, **kwargs)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.warning(message, *args, **kwargs)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.error(message, *args, **kwargs)

        def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.exception(message, *args, **kwargs)

        def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
            _logger.critical(message, *args, **kwargs)

    logger = _FallbackLogger()
