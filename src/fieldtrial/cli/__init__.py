from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name != "app":
        raise AttributeError(f"module 'fieldtrial.cli' has no attribute {name!r}")
    value = import_module("fieldtrial.cli.main").app
    globals()[name] = value
    return value
