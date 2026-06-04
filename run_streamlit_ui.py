"""Launcher for the Georgia EV Streamlit UI.

Streamlit cannot run via `python -m`, so this script execs
`streamlit run georgia_ev_intelligence/streamlit_ui/app.py` from the repo root.

Usage:
    python run_streamlit_ui.py [-- --server.port 8501 ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "georgia_ev_intelligence" / "streamlit_ui" / "app.py"
    if not app_path.exists():
        print(f"Streamlit UI not found at {app_path}", file=sys.stderr)
        return 1

    os.chdir(str(repo_root))
    extra_args = sys.argv[1:]
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *extra_args]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
