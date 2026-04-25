# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_backend_agnostic.py"
PYTHON = Path.home() / ".venvs" / "arke" / "bin" / "python"


class TestBackendAgnosticScript:
    def test_script_passes_on_repo_examples(self):
        result = subprocess.run(
            [str(PYTHON), str(SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert "PASS:" in result.stdout
        import re
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        ak_count = len(list((repo_root / "examples" / "operators").rglob("*.ak")))
        m = re.search(r"Checking (\d+) \.ak files", result.stdout)
        assert m is not None, f"unexpected stdout: {result.stdout}"
        assert int(m.group(1)) == ak_count
