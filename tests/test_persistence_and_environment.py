from __future__ import annotations

import json

import turing_eval.environment as environment
import turing_eval.transcripts as transcripts


def test_capture_environment_with_apptainer(monkeypatch, tmp_path):
    sif_path = tmp_path / "image.sif"
    sif_path.write_bytes(b"apptainer-image")
    monkeypatch.setenv("APPTAINER_CONTAINER", str(sif_path))
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    monkeypatch.setattr(environment, "_git", lambda args: "value")
    captured = environment.capture_environment()
    assert captured["git_commit"] == "value"
    assert captured["git_branch"] == "value"
    assert captured["git_dirty"] is True
    assert captured["apptainer_container_path"] == str(sif_path)
    assert captured["apptainer_container_sha256"]
    assert captured["docker_image"] == captured["apptainer_container_sha256"]


def test_capture_environment_without_container(monkeypatch):
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    monkeypatch.setattr(environment, "_git", lambda args: "")
    captured = environment.capture_environment()
    assert captured["apptainer_container_path"] is None
    assert captured["apptainer_container_sha256"] is None
    assert captured["docker_image"] is None
    assert captured["git_dirty"] is False


def test_save_transcript_json_embeds_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(transcripts, "capture_environment", lambda: {"git_commit": "abc123"})
    path = tmp_path / "trial.json"
    transcripts.save_transcript_json(
        {"trial_id": 4, "success": True, "answer": 1},
        path,
        "actor/model",
        "dist/model",
        "no_query_no_explanation",
    )
    payload = json.loads(path.read_text())
    assert payload["trial_id"] == 4
    assert payload["environment"]["git_commit"] == "abc123"
    assert payload["actor_model"] == "actor/model"


def test_write_csv_uses_superset_schema(tmp_path):
    output = tmp_path / "results.csv"
    transcripts.write_csv(
        [
            {
                "trial_id": 1,
                "success": True,
                "turns": 3,
                "answer": 0,
                "transcript_path": "trial_001.json",
                "reasoning": "because",
                "specimen_turns": 2,
            }
        ],
        output,
        "actor/model",
        "dist/model",
        "no_query_yes_explanation",
    )
    contents = output.read_text().strip().splitlines()
    assert contents[0].startswith("trial_id,actor_model,distinguisher_model,success,turns,answer,transcript_path")
    assert "no_query_yes_explanation" in contents[1]
    assert "because" in contents[1]
