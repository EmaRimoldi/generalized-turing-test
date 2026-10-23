"""Validate persisted runs before extending them; never silently discard evidence."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from .transcripts import _atomic_text


class ResumeError(ValueError):
    pass


def is_collected_trial(result: dict[str, object]) -> bool:
    return bool(result.get("success")) or result.get("error") == "distinguisher_answered_in_opening"


def implementation_digest() -> str:
    """Bind a run to its protocol, routing, backend, and declared dependencies."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "turing_eval").rglob("*.py")) + [
        root / "run_experiment.py", root / "requirements.txt", root / "requirements-hf.txt",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def run_directory(path: Path, *, resume: bool):
    """Serialize writers, including two processes explicitly resuming one run."""
    try:
        path.mkdir(parents=True, exist_ok=resume)
    except FileExistsError as exc:
        raise ResumeError(f"Run directory already exists: {path}. Use --resume or a new name.") from exc
    with (path / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResumeError(f"Another process is writing this run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResumeError(f"Unreadable run record {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeError(f"Run record must be an object: {path}")
    return payload


def _compatible_config(saved: object, current: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    mutable = {"num_trials", "max_attempts"}
    return (
        {k: v for k, v in saved.items() if k not in mutable}
        == {k: v for k, v in current.items() if k not in mutable}
        and type(saved.get("num_trials")) is int
        and saved["num_trials"] > 0
        and saved["num_trials"] <= current["num_trials"]
    )


def prepare_run(path: Path, config: dict, identity: dict) -> tuple[list[dict], int]:
    """Check every record under the caller's lock before publishing any changes."""
    manifest_path = path / "run_manifest.json"
    records = sorted(path.glob("trial_*.json")) + sorted(path.glob("failed_attempt_*.json"))
    manifest = {"schema_version": 1, "run_config": config,
                "implementation_sha256": implementation_digest(), **identity}
    if manifest_path.exists():
        saved = _read_object(manifest_path)
        if not _compatible_config(saved.get("run_config"), config) or any(
            saved.get(key) != value for key, value in manifest.items() if key != "run_config"
        ):
            raise ResumeError("Run configuration, model routes, or implementation changed; use a new run.")
    elif records or (path / "results.csv").exists():
        raise ResumeError("Historical run has no run_manifest.json; use a new run directory.")

    results = []
    seen_ids = set()
    next_attempt_id = 0
    for record_path in records:
        payload = _read_object(record_path)
        if not _compatible_config(payload.get("run_config"), config) or any(
            payload.get(key) != value for key, value in identity.items()
        ):
            raise ResumeError(f"Mismatched configuration or model identity: {record_path}")
        attempt_id = payload.get("attempt_id")
        if type(attempt_id) is not int or attempt_id < 0:
            raise ResumeError(f"Invalid attempt_id: {record_path}")
        next_attempt_id = max(next_attempt_id, attempt_id + 1)
        if record_path.name.startswith("failed_attempt_"):
            continue
        trial_id = payload.get("trial_id")
        if (
            type(trial_id) is not int or trial_id < 0 or trial_id in seen_ids
            or record_path.name != f"trial_{trial_id:03d}.json"
            or not is_collected_trial(payload)
            or type(payload.get("success")) is not bool
            or (payload["success"] and (
                type(payload.get("answer")) is not int or payload["answer"] not in {0, 1}
            ))
        ):
            raise ResumeError(f"Invalid, duplicate, or uncollected trial: {record_path}")
        seen_ids.add(trial_id)
        payload["transcript_path"] = str(record_path)
        results.append(payload)
    if len(results) > config["num_trials"]:
        raise ResumeError("Requested trial count would discard saved trials; increase --num-trials.")
    results.sort(key=lambda row: row["trial_id"])
    with _atomic_text(manifest_path) as handle:
        json.dump(manifest, handle, indent=2)
    return results, next_attempt_id
