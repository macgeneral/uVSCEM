from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path

import pytest


class _ManagerStub:
    def __init__(
        self, config_name: str, code_path: str, target_directory: str = ""
    ) -> None:
        self.config_name = config_name
        self.code_path = code_path
        self.target_directory = target_directory
        self.install_called = False
        self.initialized = False

    def install(self) -> None:
        self.install_called = True

    async def install_async(self) -> None:
        self.install_called = True

    async def initialize(self) -> None:
        self.initialized = True


def _import_extension_manager(monkeypatch: pytest.MonkeyPatch):
    if "json5" not in sys.modules:
        json5_stub = types.ModuleType("json5")
        setattr(json5_stub, "loads", lambda _value: {})
        monkeypatch.setitem(sys.modules, "json5", json5_stub)

    if "requests" not in sys.modules:
        requests_stub = types.ModuleType("requests")
        setattr(requests_stub, "Response", object)
        monkeypatch.setitem(sys.modules, "requests", requests_stub)

    if "dependency_algorithm" not in sys.modules:
        dependency_stub = types.ModuleType("dependency_algorithm")

        class _Dependencies:
            def __init__(self, _deps) -> None:
                self._deps = _deps

            def resolve_dependencies(self):
                return []

        setattr(dependency_stub, "Dependencies", _Dependencies)
        monkeypatch.setitem(sys.modules, "dependency_algorithm", dependency_stub)

    if "uvscem.api_client" not in sys.modules:
        api_stub = types.ModuleType("uvscem.api_client")

        class _CodeAPIManager:
            pass

        setattr(api_stub, "CodeAPIManager", _CodeAPIManager)
        monkeypatch.setitem(sys.modules, "uvscem.api_client", api_stub)

    if "uvscem.code_manager" not in sys.modules:
        code_stub = types.ModuleType("uvscem.code_manager")

        class _CodeManager:
            pass

        setattr(code_stub, "CodeManager", _CodeManager)
        monkeypatch.setitem(sys.modules, "uvscem.code_manager", code_stub)

    if "uvscem.extension_manager" in sys.modules:
        monkeypatch.delitem(sys.modules, "uvscem.extension_manager", raising=False)

    return importlib.import_module("uvscem.extension_manager")


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    extension_manager=None,
) -> int:
    if extension_manager is None:
        extension_manager = _import_extension_manager(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["uvscem", *args])
    try:
        extension_manager.main()
    except SystemExit as exc:  # pragma: no cover - framework dependent
        code = exc.code
        if isinstance(code, int):
            return code
        return 1
    return 0


def test_cli_help_shows_core_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, ["--help"])
    output = capsys.readouterr()
    help_text = f"{output.out}\n{output.err}"

    assert code == 0
    assert "--config-name" in help_text
    assert "--code-path" in help_text
    assert "--target-path" in help_text
    assert "--log-level" in help_text


def test_cli_invokes_manager_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_manager = _import_extension_manager(monkeypatch)
    captured: dict[str, object] = {}
    target_path = str(Path(tempfile.gettempdir()) / "extensions")

    def _factory(
        config_name: str, code_path: str, target_directory: str = ""
    ) -> _ManagerStub:
        manager = _ManagerStub(
            config_name=config_name,
            code_path=code_path,
            target_directory=target_directory,
        )
        captured["manager"] = manager
        return manager

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _factory)

    code = _run_main(
        monkeypatch,
        [
            "--config-name",
            "custom-devcontainer.json",
            "--code-path",
            "/custom/code",
            "--target-path",
            target_path,
            "--log-level",
            "info",
        ],
        extension_manager=extension_manager,
    )

    assert code == 0
    manager = captured["manager"]
    assert isinstance(manager, _ManagerStub)
    assert manager.config_name == "custom-devcontainer.json"
    assert manager.code_path == "/custom/code"
    assert manager.target_directory == target_path
    assert manager.initialized is True
    assert manager.install_called is True


def test_cli_reports_argument_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, ["--not-a-real-option"])
    output = capsys.readouterr()
    error_text = f"{output.out}\n{output.err}".lower()

    assert code != 0
    assert "error" in error_text
