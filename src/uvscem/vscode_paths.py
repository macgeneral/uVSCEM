from __future__ import annotations

import os
import platform
from pathlib import Path


def detect_runtime_environment() -> str:
    """Return a simple runtime classification used by path and CLI discovery."""
    explicit_environment = os.environ.get("UVSCEM_RUNTIME", "").strip().lower()
    if explicit_environment:
        return explicit_environment

    if os.environ.get("VSCODE_AGENT_FOLDER", "").strip():
        return "vscode-remote"

    if Path.home().joinpath(".vscode-server").exists():
        return "vscode-server"

    if Path.home().joinpath(".vscode-remote").exists():
        return "vscode-remote"

    return "local"


def resolve_vscode_root() -> Path:
    """Resolve the VS Code data root with Linux/devcontainer-first defaults."""
    explicit_root = os.environ.get("UVSCEM_VSCODE_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    agent_root = os.environ.get("VSCODE_AGENT_FOLDER", "").strip()
    if agent_root:
        return Path(agent_root).expanduser().resolve()

    runtime_environment = detect_runtime_environment()
    if runtime_environment == "vscode-server":
        return Path.home().joinpath(".vscode-server").resolve()
    if runtime_environment == "vscode-remote":
        return Path.home().joinpath(".vscode-remote").resolve()

    home = Path.home()
    candidates = [
        home.joinpath(".vscode-server"),
        home.joinpath(".vscode-remote"),
        home.joinpath(".vscode"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if platform.system().lower() == "linux":
        return home.joinpath(".vscode-server").resolve()
    return home.joinpath(".vscode").resolve()
