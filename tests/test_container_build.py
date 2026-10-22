"""Exercise build failure/retry behavior without a container runtime or downloads."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("hf", [False, True])
@pytest.mark.parametrize("codes,expected", [("0", 0), ("1,0", 0), ("1,1", 1), ("130", 130)])
def test_container_build_preserves_images_and_handles_interrupts(tmp_path, hf, codes, expected):
    repo = tmp_path / "project with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / "apptainer").mkdir()
    for name in ("build-container.sh", "build-container-hf.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    image_name = "turing-eval-hf" if hf else "turing-eval"
    store = tmp_path / "image store"
    store.mkdir()
    image = store / f"{image_name}.sif"
    image.write_text("previous image")
    link = repo / "apptainer" / image.name
    link.symlink_to(image)
    (repo / "apptainer" / f"{image_name}.def").write_text("test definition")
    binary = tmp_path / "apptainer"
    binary.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
log=Path(os.environ['BUILD_LOG'])
old=log.read_text().splitlines() if log.exists() else []
with log.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
Path(sys.argv[-2]).write_text('new image')
sys.exit(int(os.environ['BUILD_CODES'].split(',')[len(old)]))
""")
    binary.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "APPTAINER_IMAGE_DIR": str(store),
        "BUILD_LOG": str(log),
        "BUILD_CODES": codes,
    }
    entrypoint = "build-container-hf.sh" if hf else "build-container.sh"
    result = subprocess.run(
        ["bash", str(repo / "scripts" / entrypoint)], env=env, capture_output=True, text=True
    )
    assert result.returncode == expected, result.stderr
    assert image.read_text() == ("new image" if expected == 0 else "previous image")
    assert link.resolve() == image
    assert not list(store.glob("*.building.*"))
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == len(codes.split(","))
    assert "--fakeroot" in calls[0]
    if len(calls) > 1:
        assert "--fakeroot" not in calls[1]
    assert all(call[-1] == f"apptainer/{image_name}.def" for call in calls)
