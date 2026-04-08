"""Capture environment metadata for reproducibility."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Dict, Optional


def _git(args: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git"] + args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def capture_environment() -> Dict[str, object]:
    sif_sha = None
    sif_path = os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER")
    if sif_path and os.path.exists(sif_path):
        try:
            import hashlib

            hasher = hashlib.sha256()
            with open(sif_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 16), b""):
                    hasher.update(chunk)
            sif_sha = hasher.hexdigest()
        except Exception:
            sif_sha = None

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "apptainer_container_path": sif_path,
        "apptainer_container_sha256": sif_sha,
        "docker_image": sif_sha,
        "hostname": platform.node(),
    }
