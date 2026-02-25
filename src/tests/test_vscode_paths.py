from __future__ import annotations

from pathlib import Path

import pytest

from uvscem import vscode_paths


@pytest.fixture
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_detect_runtime_environment_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UVSCEM_RUNTIME", "custom-env")

    assert vscode_paths.detect_runtime_environment() == "custom-env"


def test_detect_runtime_environment_uses_vscode_agent_folder(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_RUNTIME", raising=False)
    monkeypatch.setenv("VSCODE_AGENT_FOLDER", "/tmp/vscode-agent")

    assert vscode_paths.detect_runtime_environment() == "vscode-remote"


def test_detect_runtime_environment_prefers_vscode_server_home(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_RUNTIME", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    _home.joinpath(".vscode-server").mkdir(parents=True)

    assert vscode_paths.detect_runtime_environment() == "vscode-server"


def test_detect_runtime_environment_prefers_vscode_remote_home(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_RUNTIME", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    _home.joinpath(".vscode-remote").mkdir(parents=True)

    assert vscode_paths.detect_runtime_environment() == "vscode-remote"


def test_detect_runtime_environment_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_RUNTIME", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)

    assert vscode_paths.detect_runtime_environment() == "local"


def test_resolve_vscode_root_uses_explicit_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UVSCEM_VSCODE_ROOT", "/tmp/custom-vscode")

    assert vscode_paths.resolve_vscode_root() == Path("/tmp/custom-vscode")


def test_resolve_vscode_root_uses_vscode_agent_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.setenv("VSCODE_AGENT_FOLDER", "/tmp/vscode-agent")

    assert vscode_paths.resolve_vscode_root() == Path("/tmp/vscode-agent")


def test_resolve_vscode_root_uses_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    monkeypatch.setattr(
        vscode_paths, "detect_runtime_environment", lambda: "vscode-server"
    )

    assert vscode_paths.resolve_vscode_root() == _home.joinpath(".vscode-server")


def test_resolve_vscode_root_uses_runtime_environment_remote(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    monkeypatch.setattr(
        vscode_paths, "detect_runtime_environment", lambda: "vscode-remote"
    )

    assert vscode_paths.resolve_vscode_root() == _home.joinpath(".vscode-remote")


def test_resolve_vscode_root_uses_existing_candidate(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    monkeypatch.setattr(vscode_paths, "detect_runtime_environment", lambda: "local")
    _home.joinpath(".vscode").mkdir(parents=True)

    assert vscode_paths.resolve_vscode_root() == _home.joinpath(".vscode")


def test_resolve_vscode_root_linux_fallback(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    monkeypatch.setattr(vscode_paths, "detect_runtime_environment", lambda: "local")
    monkeypatch.setattr(vscode_paths.platform, "system", lambda: "Linux")

    assert vscode_paths.resolve_vscode_root() == _home.joinpath(".vscode-server")


def test_resolve_vscode_root_non_linux_fallback(
    monkeypatch: pytest.MonkeyPatch,
    _home: Path,
) -> None:
    monkeypatch.delenv("UVSCEM_VSCODE_ROOT", raising=False)
    monkeypatch.delenv("VSCODE_AGENT_FOLDER", raising=False)
    monkeypatch.setattr(vscode_paths, "detect_runtime_environment", lambda: "local")
    monkeypatch.setattr(vscode_paths.platform, "system", lambda: "Darwin")

    assert vscode_paths.resolve_vscode_root() == _home.joinpath(".vscode")
