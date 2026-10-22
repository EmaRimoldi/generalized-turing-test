"""Ensure shell-special characters in paths cannot alter a detached batch command."""

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_tmux_launcher_quotes_caller_paths_and_uses_selected_python(tmp_path):
    repo = tmp_path / "project's folder"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_batch_in_tmux.sh", repo / "scripts/run_batch_in_tmux.sh")
    csv = tmp_path / "batch$(touch injected).csv"
    csv.write_text("test-only schedule")
    python = tmp_path / "fake-python"
    python.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
Path(os.environ['CALL_LOG']).write_text(json.dumps(sys.argv[1:]))
""")
    python.chmod(0o755)
    tmux = tmp_path / "tmux"
    tmux.write_text("""#!/usr/bin/env python3
import subprocess,sys
if sys.argv[1]=='has-session':sys.exit(1)
sys.exit(subprocess.run(sys.argv[5:],check=False).returncode)
""")
    tmux.chmod(0o755)
    log = tmp_path / "call.json"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "PYTHON": str(python),
        "CALL_LOG": str(log),
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/run_batch_in_tmux.sh"),
            csv.name,
            "output's folder",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    assert args[args.index("--csv") + 1] == str(csv)
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "output's folder")
    assert not (tmp_path / "injected").exists()
    assert not (repo / "injected").exists()
