"""Configuration files (YAML) and paths.

Strings of the form ``${NAME:-default}`` are replaced by the environment variable NAME, or by the default.
Relative paths are interpreted relative to the current working directory (run the scripts from the
repository root).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value):
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration file and expand ``${ENV:-default}`` strings."""
    with open(path, "r", encoding="utf-8") as fh:
        return _expand(yaml.safe_load(fh) or {})


def as_range(spec) -> list[int]:
    """``[start, stop, step]`` -> ``list(range(start, stop, step))``; a plain list is returned unchanged."""
    if isinstance(spec, dict):
        return list(range(spec["start"], spec["stop"], spec.get("step", 1)))
    return list(spec)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
