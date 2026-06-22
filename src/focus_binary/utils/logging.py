from __future__ import annotations

import logging
import os
from typing import Optional


LOG_LEVEL = os.environ.get("FOCUS_LOG_LEVEL", "INFO").upper()


def _configure_root():
    root = logging.getLogger("focus_binary")
    if root.handlers:
        return root
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    base = _configure_root()
    return base.getChild(name) if name else base
