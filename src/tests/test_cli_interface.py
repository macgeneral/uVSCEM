from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path

import pytest


class _ManagerStub:
    def __init__(
        self,
        config_name: str,
        code_path: str,
        target_directory: str = "",
        allow_unsigned: bool = False,
        allow_untrusted_urls: bool = False,
        allow_http: bool = False,
        disable_ssl_verification: bool = False,
        ca_bundle: str = "",
    ) -> None:
        self.config_name = config_name
        self.code_path = code_path
        self.target_directory = target_directory
        self.allow_unsigned = allow_unsigned
        self.allow_untrusted_urls = allow_untrusted_urls
        self.allow_http = allow_http
        self.disable_ssl_verification = disable_ssl_verification
        self.ca_bundle = ca_bundle
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
        json5_stub.loads = lambda _value: {}  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "json5", json5_stub)

    if "requests" not in sys.modules:
        requests_stub = types.ModuleType("requests")
        requests_stub.Response = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "requests", requests_stub)

    if "dependency_algorithm" not in sys.modules:
        dependency_stub = types.ModuleType("dependency_algorithm")

        class _Dependencies:
            def __init__(self, _deps) -> None:
                self._deps = _deps

            def resolve_dependencies(self):
                return []

        dependency_stub.Dependencies = _Dependencies  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "dependency_algorithm", dependency_stub)

    if "uvscem.api_client" not in sys.modules:
        api_stub = types.ModuleType("uvscem.api_client")

        class _CodeAPIManager:
            pass

        api_stub.CodeAPIManager = _CodeAPIManager  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "uvscem.api_client", api_stub)

    if "uvscem.code_manager" not in sys.modules:
        code_stub = types.ModuleType("uvscem.code_manager")

        class _CodeManager:
            pass

        code_stub.CodeManager = _CodeManager  # type: ignore[attr-defined]
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
    assert "--allow-untrusted-urls" in help_text
    assert "--disable-ssl-verification" in help_text
    assert "INSECURE" in help_text


def test_cli_invokes_manager_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extension_manager = _import_extension_manager(monkeypatch)
    captured: dict[str, object] = {}
    code_path = str(Path(tempfile.gettempdir()) / "custom" / "code")
    target_path = str(Path(tempfile.gettempdir()) / "extensions")
    config_file = tmp_path / "custom-devcontainer.json"
    config_file.write_text("{}", encoding="utf-8")

    def _factory(
        config_name: str,
        code_path: str,
        target_directory: str = "",
        allow_unsigned: bool = False,
        allow_untrusted_urls: bool = False,
        allow_http: bool = False,
        disable_ssl_verification: bool = False,
        ca_bundle: str = "",
    ) -> _ManagerStub:
        manager = _ManagerStub(
            config_name=config_name,
            code_path=code_path,
            target_directory=target_directory,
            allow_unsigned=allow_unsigned,
            allow_untrusted_urls=allow_untrusted_urls,
            allow_http=allow_http,
            disable_ssl_verification=disable_ssl_verification,
            ca_bundle=ca_bundle,
        )
        captured["manager"] = manager
        return manager

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _factory)

    code = _run_main(
        monkeypatch,
        [
            "--config-name",
            str(config_file),
            "--code-path",
            code_path,
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
    assert manager.config_name == str(config_file)
    assert manager.code_path == code_path
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


def test_main_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    extension_manager = _import_extension_manager(monkeypatch)
    called: list[bool] = []
    monkeypatch.setattr(extension_manager, "main", lambda: called.append(True))
    runpy.run_module("uvscem", run_name="__main__", alter_sys=False)
    assert called == [True]


def test_install_missing_config_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with caplog.at_level(logging.ERROR):
        code = _run_main(
            monkeypatch,
            ["install", "--config-name", "does-not-exist.json"],
        )

    assert code == 1
    assert any("not found" in record.message.lower() for record in caplog.records)


def test_export_missing_config_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with caplog.at_level(logging.ERROR):
        code = _run_main(
            monkeypatch,
            ["export", "--config-name", "does-not-exist.json"],
        )

    assert code == 1
    assert any("not found" in record.message.lower() for record in caplog.records)


def test_version_flag_prints_version_and_user_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, ["--version"])
    output = capsys.readouterr()
    text = f"{output.out}\n{output.err}"

    assert code == 0
    assert "uvscem" in text
    assert "User-Agent:" in text


def test_import_missing_bundle_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import logging

    missing_bundle = str(tmp_path / "no-such-bundle")
    with caplog.at_level(logging.ERROR):
        code = _run_main(
            monkeypatch,
            ["import", "--bundle-path", missing_bundle],
        )

    assert code == 1
    assert any("not found" in record.message.lower() for record in caplog.records)
