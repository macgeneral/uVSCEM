from __future__ import annotations

import builtins
import os
import socket
from pathlib import Path

import pytest

from uvscem import code_manager
from uvscem.code_manager import CodeManager


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

    manager.find_socket(update_environment=True)

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

    manager.find_socket(update_environment=True)

    assert manager.socket_path is None
    assert "VSCODE_IPC_HOOK_CLI" not in os.environ


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

    manager.find_latest_code(update_environment=True)

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
    manager.find_latest_code(update_environment=True)

    path_items = os.environ["PATH"].split(":")
    assert path_items[0] == expected
    assert path_items.count(expected) == 1


def test_find_latest_code_handles_missing_or_invalid_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "vscode"
    manager = CodeManager.__new__(CodeManager)
    manager.socket_path = None
    manager.code_path = None

    monkeypatch.setattr(code_manager, "vscode_root", root)

    with caplog.at_level("WARNING"):
        manager.find_latest_code(update_environment=True)

    assert "No VSCode remote CLI executable found" in caplog.text

    _write_code_launcher(
        root / "bin" / "commit-a" / "bin" / "remote-cli" / "code", "1.0.0", "commit-a"
    )
    monkeypatch.setattr(builtins, "max", lambda _values: "missing")

    with caplog.at_level("WARNING"):
        manager.find_latest_code(update_environment=True)

    assert "Unable to determine latest VSCode version" in caplog.text


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
