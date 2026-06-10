"""统一日志：所有模块通过 get_logger 获得带时间戳的 logger，便于调试。"""
from __future__ import annotations

import logging
import sys

_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DATEFMT = "%H:%M:%S"
_INITED = False


def _init_root() -> None:
    global _INITED
    if _INITED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root = logging.getLogger("cam")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _INITED = True


def get_logger(name: str = "cam") -> logging.Logger:
    """返回 'cam.<name>' 命名的 logger。"""
    _init_root()
    return logging.getLogger(name if name.startswith("cam") else f"cam.{name}")


def set_level(level: str | int) -> None:
    """全局调整日志等级，例如 'DEBUG' 用于深入调试。"""
    _init_root()
    logging.getLogger("cam").setLevel(level)
