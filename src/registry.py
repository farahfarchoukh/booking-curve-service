"""
Minimal model registry: versioned artifact directories + a "current"
pointer, so promoting or rolling back a model is a file write, not a code
change or a redeploy of a different image.

This exists because the first pass of this submission saved every model to
the same `artifacts/model/` path, in place — meaning there was no way to
answer "what was serving yesterday" or roll back to it, despite DESIGN.md
§6.5 explicitly describing a rollback signal. That was a real gap between
the design doc and the code; this closes it.

Layout:
    artifacts/model/
      current.json          -> {"version": "<version>"}
      20260906T142301Z/      one versioned artifact (level.txt, shape.txt, ...)
      20260905T090000Z/
      ...

`train.py` calls `new_version()` + `save_pointer()` after a successful
save. `predict.py` / `api.py` call `resolve_model_dir()`, which honors an
explicit version (env var or CLI flag — pinning for rollback or canary)
before falling back to `current.json`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

VERSION_ENV_VAR = "BOOKING_CURVE_MODEL_VERSION"


def new_version(explicit: str | None = None) -> str:
    """A sortable, unique version id. `explicit` lets CI stamp a git SHA
    or release tag instead of a timestamp — pass it as
    `MODEL_VERSION=<sha> python -m src.train` and it flows through here."""
    return explicit or os.environ.get("MODEL_VERSION") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def save_pointer(base_dir: Path, version: str) -> None:
    """Atomically point `current.json` at `version`. Write-to-temp-then-
    rename is atomic on both POSIX and Windows as long as source and
    destination are on the same filesystem, which they always are here —
    this is what stops a reader from ever seeing a half-written pointer."""
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp = base_dir / "current.json.tmp"
    tmp.write_text(json.dumps({"version": version}, indent=2))
    tmp.replace(base_dir / "current.json")


def current_version(base_dir: Path) -> str:
    pointer = base_dir / "current.json"
    if not pointer.exists():
        raise FileNotFoundError(
            f"No current.json in {base_dir} — has `python -m src.train` been run?"
        )
    return json.loads(pointer.read_text())["version"]


def resolve_model_dir(base_dir: Path, version: str | None = None) -> Path:
    """`version` (explicit arg, or the BOOKING_CURVE_MODEL_VERSION env var
    if arg is None) pins a specific model — this is the rollback lever:
    point it at an older version directory and restart, no rebuild. With
    neither set, resolves through `current.json`, i.e. "whatever the last
    successful train run promoted."
    """
    pinned = version or os.environ.get(VERSION_ENV_VAR)
    resolved = pinned or current_version(base_dir)
    path = base_dir / resolved
    if not path.exists():
        raise FileNotFoundError(f"Model version '{resolved}' not found under {base_dir}")
    return path


def list_versions(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        return []
    return sorted(p.name for p in base_dir.iterdir() if p.is_dir())
