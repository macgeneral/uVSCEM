from __future__ import annotations

import asyncio
import builtins
import os
import socket
from pathlib import Path

import pytest

from uvscem import code_manager
from uvscem.code_manager import CodeManager


def test_init_calls_socket_and_code_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = CodeManager()

    assert manager.socket_path is None
    assert manager.code_path is None


def test_find_socket_removes_stale_and_sets_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "vscode-ipc-stale.sock"
    active = tmp_path / "vscode-ipc-active.sock"
    stale.write_text("stale")
    active.write_text("active")

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        CodeManager,
        "is_socket_closed",
        staticmethod(lambda path: path.name.endswith("stale.sock")),
    )

    asyncio.run(manager.find_socket(update_environment=True))

    assert not stale.exists()
    assert manager.socket_path == active
    assert os.environ["VSCODE_IPC_HOOK_CLI"] == str(active)


def test_find_socket_without_any_socket_keeps_environment_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        CodeManager,
        "is_socket_closed",
        staticmethod(lambda _path: False),
    )
    monkeypatch.delenv("VSCODE_IPC_HOOK_CLI", raising=False)

    asyncio.run(manager.find_socket(update_environment=True))

    assert manager.socket_path is None
    assert "VSCODE_IPC_HOOK_CLI" not in os.environ


def test_find_socket_keeps_first_active_when_multiple_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "vscode-ipc-aaa.sock"
    second = tmp_path / "vscode-ipc-bbb.sock"
    first.write_text("a")
    second.write_text("b")

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        CodeManager, "is_socket_closed", staticmethod(lambda _path: False)
    )

    asyncio.run(manager.find_socket(update_environment=False))

    assert manager.socket_path == first


def _write_code_launcher(script_path: Path, version: str, commit: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(f"VERSION={version}\nCOMMIT={commit}\n")


def test_find_latest_code_updates_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "vscode"
    _write_code_launcher(
        root / "bin" / "commit1" / "bin" / "remote-cli" / "code", "1.0.0", "commit1"
    )
    _write_code_launcher(
        root / "bin" / "commit2" / "bin" / "remote-cli" / "code", "2.0.0", "commit2"
    )

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setattr(code_manager, "vscode_root", root)
    monkeypatch.setenv("PATH", "")

    asyncio.run(manager.find_latest_code(update_environment=True))

    expected = str(root / "bin" / "commit2" / "bin" / "remote-cli")
    assert str(manager.code_path) == expected
    assert os.environ["PATH"].split(":")[0] == expected


def test_find_latest_code_reorders_when_path_has_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "vscode"
    _write_code_launcher(
        root / "bin" / "commit9" / "bin" / "remote-cli" / "code", "9.0.0", "commit9"
    )

    expected = str(root / "bin" / "commit9" / "bin" / "remote-cli")
    monkeypatch.setenv("PATH", f"alpha:{expected}:beta:{expected}")
    monkeypatch.setattr(code_manager, "vscode_root", root)

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None
    asyncio.run(manager.find_latest_code(update_environment=True))

    path_items = os.environ["PATH"].split(":")
    assert path_items[0] == expected
    assert path_items.count(expected) == 1


def test_find_latest_code_without_environment_update_keeps_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "vscode"
    _write_code_launcher(
        root / "bin" / "commit3" / "bin" / "remote-cli" / "code", "3.0.0", "commit3"
    )

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setattr(code_manager, "vscode_root", root)
    monkeypatch.setenv("PATH", "alpha:beta")

    asyncio.run(manager.find_latest_code(update_environment=False))

    assert str(manager.code_path).endswith("commit3/bin/remote-cli")
    assert os.environ["PATH"] == "alpha:beta"


def test_find_latest_code_handles_missing_or_invalid_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "vscode"
    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setattr(code_manager, "vscode_root", root)

    with caplog.at_level("WARNING"):
        asyncio.run(manager.find_latest_code(update_environment=True))

    assert "No VSCode remote CLI executable found" in caplog.text

    _write_code_launcher(
        root / "bin" / "commit-a" / "bin" / "remote-cli" / "code", "1.0.0", "commit-a"
    )
    original_max = builtins.max

    def _max(values, *args, **kwargs):
        if not args and not kwargs:
            return "missing"
        return original_max(values, *args, **kwargs)

    monkeypatch.setattr(builtins, "max", _max)

    with caplog.at_level("WARNING"):
        asyncio.run(manager.find_latest_code(update_environment=True))

    assert "Unable to determine latest VSCode version" in caplog.text


def test_find_latest_code_ignores_launcher_missing_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "vscode"
    script = root / "bin" / "commit-x" / "bin" / "remote-cli" / "code"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("VERSION=1.0.0\n")

    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setattr(code_manager, "vscode_root", root)

    with caplog.at_level("WARNING"):
        asyncio.run(manager.find_latest_code(update_environment=False))

    assert "No VSCode remote CLI executable found" in caplog.text


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BlockingIOError(), False),
        (ConnectionResetError(), True),
        (ConnectionRefusedError(), True),
        (RuntimeError("unexpected"), False),
    ],
)
def test_is_socket_closed_branches(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: bool,
) -> None:
    class _FakeSocket:
        def connect(self, _value: str) -> None:
            return None

        def recv(self, _size: int, _flags: int) -> None:
            raise error

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: _FakeSocket())

    assert CodeManager.is_socket_closed(Path("/tmp/fake.sock")) is expected


def test_is_socket_closed_returns_false_when_recv_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSocket:
        def connect(self, _value: str) -> None:
            return None

        def recv(self, _size: int, _flags: int) -> bytes:
            return b"ok"

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: _FakeSocket())

    assert CodeManager.is_socket_closed(Path("/tmp/fake.sock")) is False


def test_async_code_manager_methods_delegate_to_private_sync_methods() -> None:
    manager = CodeManager.__new__(CodeManager)
    calls: list[str] = []

    manager._find_socket_sync = lambda update_environment=False: calls.append(
        f"socket:{update_environment}"
    )
    manager._find_latest_code_sync = lambda update_environment=False: calls.append(
        f"code:{update_environment}"
    )

    asyncio.run(manager.find_socket(update_environment=True))
    asyncio.run(manager.find_latest_code(update_environment=False))
    asyncio.run(manager.initialize())

    assert calls == ["socket:True", "code:False", "socket:True", "code:True"]
