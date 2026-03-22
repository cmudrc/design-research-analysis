"""Version helpers for the installed package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("design-research-analysis")
except PackageNotFoundError:
    __version__ = "0+unknown"
