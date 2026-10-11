"""Settings read from the environment. A variable left empty counts as not set."""

import os
from pathlib import Path


def text(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def whole(name: str, default: int) -> int:
    value = text(name)
    return int(value) if value.isdigit() else default


def secret(name: str) -> str:
    """A secret given directly, or in the file `<name>_FILE` names (a Docker secret volume)."""
    direct = text(name)
    if direct:
        return direct
    path = text(f"{name}_FILE")
    try:
        return Path(path).read_text().strip() if path else ""
    except OSError:
        return ""
