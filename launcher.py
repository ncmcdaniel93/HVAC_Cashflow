"""Windows launcher entrypoint for packaged Streamlit app."""

from __future__ import annotations

import os
import pathlib
import sys


def _bundle_root() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(getattr(sys, "_MEIPASS"))
    return pathlib.Path(__file__).resolve().parent


def _runtime_root() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent


def main() -> None:
    bundle_root = _bundle_root()
    runtime_root = _runtime_root()
    app_path = bundle_root / "app.py"

    # Keep local persistence (.local_store) beside the executable.
    os.chdir(runtime_root)

    # Disable usage stats prompt in distributed builds.
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
