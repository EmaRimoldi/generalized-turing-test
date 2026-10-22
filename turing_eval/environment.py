"""Capture environment metadata for reproducibility."""
from __future__ import annotations

import os
import hashlib
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


def _git(args: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git"] + args, cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


@lru_cache(maxsize=8)
def _container_digest(path: str, identity: tuple[int, ...]) -> str:
    """Hash an unchanged container once, rather than once per trial."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def capture_environment() -> Dict[str, object]:
    sif_sha = None
    sif_path = os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER")
    if sif_path and os.path.exists(sif_path):
        try:
            image = Path(sif_path).resolve()
            stat = image.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            sif_sha = _container_digest(str(image), identity)
        except Exception:
            sif_sha = None

    dirty = _git(["status", "--porcelain"])
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "apptainer_container_path": sif_path,
        "apptainer_container_sha256": sif_sha,
        "docker_image": sif_sha,
        "hostname": platform.node(),
    }
