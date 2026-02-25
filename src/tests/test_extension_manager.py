from __future__ import annotations

import asyncio
import json
import runpy
import subprocess
import sys
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from uvscem import extension_manager
from uvscem.extension_manager import CodeExtensionManager


@pytest.fixture
def bare_manager(tmp_path: Path) -> CodeExtensionManager:
    async def _noop_socket(update_environment: bool = False) -> None:
        return None

    manager = CodeExtensionManager.__new__(CodeExtensionManager)
    manager.extension_dependencies = defaultdict(set)
    manager.extension_metadata = {}
    manager.api_manager = SimpleNamespace()
    manager.socket_manager = SimpleNamespace(find_socket=_noop_socket)
    manager.code_binary = "code"
    manager.dev_container_config_path = tmp_path / "devcontainer.json"
    manager.extensions = []
    manager.installed = []
    manager.target_path = tmp_path / "cache"
    manager.target_path.mkdir(parents=True, exist_ok=True)
    return manager


async def _find_installed() -> list[dict]:
    return []


async def _find_socket_collector(
    socket_calls: list[bool], update_environment: bool = False
) -> None:
    socket_calls.append(update_environment)


def test_dependency_helpers(bare_manager: CodeExtensionManager) -> None:
    assert bare_manager.sanitize_dependencies(
        ["vscode.python", "ms-python.python"]
    ) == ["ms-python.python"]

    extensions = ["one"]
    bare_manager.add_missing_dependency(extensions, ["one", "two"])
    assert extensions == ["one", "two"]


def test_init_sets_paths_and_creates_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "devcontainer.json"
    config_path.write_text("{}")
    target_dir = tmp_path / "target-cache"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(extension_manager, "CodeAPIManager", lambda: SimpleNamespace())
    monkeypatch.setattr(extension_manager, "CodeManager", lambda: SimpleNamespace())
    manager = CodeExtensionManager(
        config_name="devcontainer.json",
        code_path="/usr/bin/code",
        target_directory=str(target_dir),
    )

    assert manager.code_binary == "/usr/bin/code"
    assert manager.dev_container_config_path == config_path.resolve()
    assert manager.extensions == []
    assert manager.installed == []
    assert manager.target_path == target_dir.resolve()
    assert manager.target_path.is_dir()


def test_init_accepts_absolute_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "absolute-devcontainer.json"
    config_path.write_text("{}")

    monkeypatch.setattr(extension_manager, "CodeAPIManager", lambda: SimpleNamespace())
    monkeypatch.setattr(extension_manager, "CodeManager", lambda: SimpleNamespace())
    manager = CodeExtensionManager(config_name=str(config_path), code_path="code")

    assert manager.dev_container_config_path == config_path


def test_parse_all_extensions_collects_metadata_and_dependencies(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.dev_container_config_path.write_text("{}")

    monkeypatch.setattr(
        extension_manager.json5,
        "loads",
        lambda _value: {
            "customizations": {
                "vscode": {"extensions": ["publisher.alpha", "publisher.skip"]}
            }
        },
    )

    metadata_map = {
        "publisher.alpha": [
            {
                "version": "1.0.0",
                "dependencies": ["dep.one", "vscode.internal"],
                "extension_pack": ["pack.one"],
            }
        ],
        "publisher.skip": [],
        "dep.one": [{"version": "1.0.0", "dependencies": [], "extension_pack": []}],
        "pack.one": [{"version": "1.0.0", "dependencies": [], "extension_pack": []}],
    }

    async def _metadata(extension_id: str):
        return {extension_id: metadata_map[extension_id]}

    bare_manager.api_manager = cast(
        Any, SimpleNamespace(get_extension_metadata=_metadata)
    )

    class _Dependencies:
        def __init__(self, deps):
            self.deps = deps

        def resolve_dependencies(self):
            return ["dep.one", "pack.one", "publisher.alpha"]

    monkeypatch.setattr(extension_manager, "Dependencies", _Dependencies)

    resolved = asyncio.run(bare_manager.parse_all_extensions())

    assert resolved == ["dep.one", "pack.one", "publisher.alpha"]
    assert bare_manager.extension_dependencies["publisher.alpha"] == {"dep.one"}
    assert bare_manager.extension_metadata["publisher.alpha"]["version"] == "1.0.0"


def test_install_async_iterates_extensions_and_sleeps(
    bare_manager: CodeExtensionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare_manager.extensions = ["one", "two"]
    called: list[str] = []

    async def _exclude() -> None:
        called.append("exclude")

    async def _install(
        ext: str, extension_pack: bool = False, retries: int = 0
    ) -> None:
        called.append(f"install:{ext}")

    async def _sleep(_seconds: int) -> None:
        called.append("sleep")

    monkeypatch.setattr(bare_manager, "exclude_installed", _exclude)
    monkeypatch.setattr(bare_manager, "install_extension", _install)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _sleep)

    asyncio.run(bare_manager.install_async())

    assert called == ["exclude", "install:one", "sleep", "install:two", "sleep"]


def test_filename_helpers_use_extension_metadata(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {"pub.ext": {"version": "2.3.4"}}

    assert bare_manager.get_dirname("pub.ext") == "pub.ext-2.3.4"
    assert bare_manager.get_filename("pub.ext") == "pub.ext-2.3.4.vsix"


def test_download_extension_writes_and_moves_file(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "url": "https://download/file.vsix"}
    }

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            assert chunk_size == 1024 * 8
            yield b"abc"
            yield b""
            yield b"def"

    bare_manager.api_manager = cast(
        Any,
        SimpleNamespace(
            session=SimpleNamespace(get=lambda *args, **kwargs: _Response())
        ),
    )

    downloaded = asyncio.run(bare_manager.download_extension("pub.ext"))

    assert downloaded.is_file()
    assert downloaded.read_bytes() == b"abcdef"


def test_download_extension_fails_without_url(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {"pub.ext": {"version": "1.0.0", "url": ""}}

    with pytest.raises(ValueError):
        asyncio.run(bare_manager.download_extension("pub.ext"))


def test_download_signature_archive_writes_and_moves_file(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "signature": "https://download/file.sigzip"}
    }

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            assert chunk_size == 1024 * 8
            yield b"sig"
            yield b""
            yield b"zip"

    bare_manager.api_manager = cast(
        Any,
        SimpleNamespace(
            session=SimpleNamespace(get=lambda *args, **kwargs: _Response())
        ),
    )

    downloaded = asyncio.run(bare_manager.download_signature_archive("pub.ext"))

    assert downloaded.is_file()
    assert downloaded.read_bytes() == b"sigzip"


def test_download_signature_archive_fails_without_url(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {"pub.ext": {"version": "1.0.0", "signature": ""}}

    with pytest.raises(ValueError):
        asyncio.run(bare_manager.download_signature_archive("pub.ext"))


def test_verify_extension_signature_runs_vsce_sign(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vsix_path = tmp_path / "ext.vsix"
    sig_path = tmp_path / "ext.sigzip"
    vsix_path.write_text("payload")
    sig_path.write_text("payload")
    bare_manager.vsce_sign_binary = Path("/tmp/vsce-sign")

    called: dict[str, list[str]] = {}

    def _run(cmd, capture_output, check, text):
        called["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(extension_manager.subprocess, "run", _run)

    asyncio.run(bare_manager.verify_extension_signature(vsix_path, sig_path))

    assert called["cmd"][0] == "/tmp/vsce-sign"


def test_verify_extension_signature_raises_when_not_initialized(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
) -> None:
    vsix_path = tmp_path / "ext.vsix"
    sig_path = tmp_path / "ext.sigzip"
    vsix_path.write_text("payload")
    sig_path.write_text("payload")
    bare_manager.vsce_sign_binary = None

    with pytest.raises(RuntimeError):
        asyncio.run(bare_manager.verify_extension_signature(vsix_path, sig_path))


def test_verify_extension_signature_raises_on_nonzero_exit(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vsix_path = tmp_path / "ext.vsix"
    sig_path = tmp_path / "ext.sigzip"
    vsix_path.write_text("payload")
    sig_path.write_text("payload")
    bare_manager.vsce_sign_binary = Path("/tmp/vsce-sign")

    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"),
    )

    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(bare_manager.verify_extension_signature(vsix_path, sig_path))


def test_install_extension_manually_extracts_and_updates_json(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "vscode-root"
    monkeypatch.setattr(extension_manager, "vscode_root", root)
    bare_manager.extension_metadata = {"pub.ext": {"version": "1.0.0"}}

    vsix_path = tmp_path / "pub.ext-1.0.0.vsix"
    with zipfile.ZipFile(vsix_path, "w") as archive:
        archive.writestr("extension/package.json", "{}")

    called: list[str] = []

    async def _update(extension_id: str = "", extension_ids: list[str] = []):
        called.append(extension_id)

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update)

    asyncio.run(
        bare_manager.install_extension_manually("pub.ext", vsix_path, update_json=True)
    )

    extracted_file = root / "extensions" / "pub.ext-1.0.0" / "package.json"
    assert extracted_file.is_file()
    assert called == ["pub.ext"]


def test_install_extension_manually_removes_existing_target(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "vscode-root"
    monkeypatch.setattr(extension_manager, "vscode_root", root)
    bare_manager.extension_metadata = {"pub.ext": {"version": "1.0.0"}}

    existing_dir = root / "extensions" / "pub.ext-1.0.0"
    existing_dir.mkdir(parents=True, exist_ok=True)
    (existing_dir / "old.txt").write_text("old")

    vsix_path = tmp_path / "pub.ext-1.0.0.vsix"
    with zipfile.ZipFile(vsix_path, "w") as archive:
        archive.writestr("extension/new.txt", "new")

    asyncio.run(
        bare_manager.install_extension_manually("pub.ext", vsix_path, update_json=False)
    )

    assert (existing_dir / "new.txt").read_text() == "new"


def test_install_extension_manually_can_skip_json_update(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "vscode-root"
    monkeypatch.setattr(extension_manager, "vscode_root", root)
    bare_manager.extension_metadata = {"pub.ext": {"version": "1.0.0"}}

    vsix_path = tmp_path / "pub.ext-1.0.0.vsix"
    with zipfile.ZipFile(vsix_path, "w") as archive:
        archive.writestr("extension/package.json", "{}")

    called: list[str] = []

    async def _update_skip(extension_id: str = "", extension_ids: list[str] = []):
        called.append("updated")

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update_skip)

    asyncio.run(
        bare_manager.install_extension_manually("pub.ext", vsix_path, update_json=False)
    )

    assert called == []


def test_update_extensions_json_success_path(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]")
    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)

    async def _find_installed() -> list[dict]:
        return []

    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)

    bare_manager.extension_metadata = {
        "pub.ext": {"installation_metadata": {"identifier": {"id": "pub.ext"}}}
    }

    asyncio.run(
        bare_manager.update_extensions_json(extension_id="pub.ext", extension_ids=[])
    )

    assert json.loads(json_path.read_text()) == [{"identifier": {"id": "pub.ext"}}]


def test_update_extensions_json_handles_dump_error_and_recovers(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)

    original_dump = json.dump
    calls = {"count": 0}

    def _dump(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first write failed")
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(extension_manager.json, "dump", _dump)

    asyncio.run(bare_manager.update_extensions_json(extension_ids=[]))

    assert json_path.read_text() == '"[]"'


def test_update_extensions_json_skips_missing_installation_metadata(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)

    bare_manager.extension_metadata = {"pub.ext": {}}
    asyncio.run(bare_manager.update_extensions_json(extension_ids=["pub.ext"]))

    assert json.loads(json_path.read_text()) == []


def test_update_extensions_json_handles_missing_backup_path(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)

    original_dump = json.dump
    calls = {"count": 0}

    def _dump(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first write failed")
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(extension_manager.json, "dump", _dump)

    original_is_file = Path.is_file
    original_stat = Path.stat

    def _is_file(path: Path) -> bool:
        if path.name == "extensions.json.bak":
            return False
        return original_is_file(path)

    def _stat(path: Path):
        if path == json_path:
            return SimpleNamespace(st_size=0)
        return original_stat(path)

    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(Path, "stat", _stat)

    asyncio.run(bare_manager.update_extensions_json(extension_ids=[]))

    assert json_path.read_text() == '"[]"'


def test_update_extensions_json_keeps_non_empty_restored_file_on_error(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text('[{"existing": true}]')

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)
    monkeypatch.setattr(
        extension_manager.json,
        "dump",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    asyncio.run(bare_manager.update_extensions_json(extension_ids=[]))

    assert json_path.read_text() == '[{"existing": true}]'


def test_install_extension_handles_extension_pack_and_recursion(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "main.ext": {"version": "1.0.0", "extension_pack": ["child.ext"]},
        "child.ext": {"version": "1.0.0", "extension_pack": []},
    }
    bare_manager.extensions = ["child.ext"]

    main_path = bare_manager.target_path / "main.ext-1.0.0.vsix"
    child_path = bare_manager.target_path / "child.ext-1.0.0.vsix"
    main_path.write_bytes(b"main")
    child_path.write_bytes(b"child")

    updates: list[list[str]] = []
    manual: list[str] = []

    async def _update_pack(extension_id: str = "", extension_ids: list[str] = []):
        updates.append(list(extension_ids))

    async def _manual_pack(
        extension_id: str, extension_path: Path, update_json: bool = True
    ) -> None:
        manual.append(extension_id)

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update_pack)
    monkeypatch.setattr(bare_manager, "install_extension_manually", _manual_pack)

    asyncio.run(bare_manager.install_extension("main.ext"))

    assert updates == [["main.ext", "child.ext"]]
    assert manual == ["child.ext", "main.ext"]


def test_install_extension_extension_pack_when_already_in_pack_mode(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "main.ext": {"version": "1.0.0", "extension_pack": ["child.ext"]}
    }
    bare_manager.extensions = []
    main_path = bare_manager.target_path / "main.ext-1.0.0.vsix"
    main_path.write_text("payload")

    updates: list[list[str]] = []
    manual: list[str] = []

    async def _update_pack(extension_id: str = "", extension_ids: list[str] = []):
        updates.append(list(extension_ids))

    async def _manual_pack(
        extension_id: str, extension_path: Path, update_json: bool = True
    ) -> None:
        manual.append(extension_id)

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update_pack)
    monkeypatch.setattr(bare_manager, "install_extension_manually", _manual_pack)

    asyncio.run(bare_manager.install_extension("main.ext", extension_pack=True))

    assert updates == [["child.ext"]]
    assert manual == ["main.ext"]


def test_install_extension_runs_subprocess_and_retries_once(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    file_path = bare_manager.target_path / "pub.ext-1.0.0.vsix"
    file_path.write_text("payload")

    socket_calls: list[bool] = []

    async def _find_socket(update_environment: bool = False) -> None:
        socket_calls.append(update_environment)

    bare_manager.socket_manager = cast(
        Any,
        SimpleNamespace(
            find_socket=lambda update_environment=False: _find_socket_collector(
                socket_calls, update_environment
            )
        ),
    )

    attempts = {"count": 0}

    def _run(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise subprocess.CalledProcessError(1, ["code"], "", "")
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(extension_manager.subprocess, "run", _run)

    asyncio.run(bare_manager.install_extension("pub.ext"))

    assert attempts["count"] == 2
    assert socket_calls == [True]


def test_install_extension_downloads_when_file_missing(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }

    downloaded = bare_manager.target_path / "downloaded-pub.ext-1.0.0.vsix"
    downloaded.write_text("payload")

    async def _download(extension_id: str) -> Path:
        return downloaded

    monkeypatch.setattr(bare_manager, "download_extension", _download)
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="ok", stderr=""),
    )

    asyncio.run(bare_manager.install_extension("pub.ext"))


def test_install_extension_verifies_signature_when_present(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {
            "version": "1.0.0",
            "extension_pack": [],
            "signature": "https://download/file.sigzip",
        }
    }
    vsix_file = bare_manager.target_path / "pub.ext-1.0.0.vsix"
    sig_file = bare_manager.target_path / "pub.ext-1.0.0.sigzip"
    vsix_file.write_text("payload")
    sig_file.write_text("signature")
    bare_manager.vsce_sign_binary = Path("/tmp/vsce-sign")

    called: list[str] = []

    async def _verify(extension_path: Path, signature_path: Path) -> None:
        called.append(f"verify:{extension_path.name}:{signature_path.name}")

    monkeypatch.setattr(bare_manager, "verify_extension_signature", _verify)
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="ok", stderr=""),
    )

    asyncio.run(bare_manager.install_extension("pub.ext"))

    assert called == ["verify:pub.ext-1.0.0.vsix:pub.ext-1.0.0.sigzip"]


def test_install_extension_downloads_missing_signature_archive_before_verify(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {
            "version": "1.0.0",
            "extension_pack": [],
            "signature": "https://download/file.sigzip",
        }
    }
    vsix_file = bare_manager.target_path / "pub.ext-1.0.0.vsix"
    vsix_file.write_text("payload")
    bare_manager.vsce_sign_binary = Path("/tmp/vsce-sign")

    called: list[str] = []

    async def _download_sig(extension_id: str) -> Path:
        called.append(f"download:{extension_id}")
        path = bare_manager.target_path / "pub.ext-1.0.0.sigzip"
        path.write_text("signature")
        return path

    async def _verify(extension_path: Path, signature_path: Path) -> None:
        called.append(f"verify:{signature_path.name}")

    monkeypatch.setattr(bare_manager, "download_signature_archive", _download_sig)
    monkeypatch.setattr(bare_manager, "verify_extension_signature", _verify)
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="ok", stderr=""),
    )

    asyncio.run(bare_manager.install_extension("pub.ext"))

    assert called == ["download:pub.ext", "verify:pub.ext-1.0.0.sigzip"]


def test_install_extension_treats_error_text_as_failure_and_stops_at_max_retries(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    file_path = bare_manager.target_path / "pub.ext-1.0.0.vsix"
    file_path.write_text("payload")

    socket_calls: list[bool] = []
    bare_manager.socket_manager = cast(
        Any,
        SimpleNamespace(
            find_socket=lambda update_environment=False: _find_socket_collector(
                socket_calls, update_environment
            )
        ),
    )

    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="Error: failure", stderr=""),
    )

    asyncio.run(
        bare_manager.install_extension("pub.ext", retries=extension_manager.max_retries)
    )

    assert socket_calls == [True]


def test_install_extension_treats_stderr_error_as_failure(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    file_path = bare_manager.target_path / "pub.ext-1.0.0.vsix"
    file_path.write_text("payload")

    socket_calls: list[bool] = []
    bare_manager.socket_manager = cast(
        Any,
        SimpleNamespace(
            find_socket=lambda update_environment=False: _find_socket_collector(
                socket_calls, update_environment
            )
        ),
    )

    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", stderr="Error: boom"),
    )

    asyncio.run(
        bare_manager.install_extension("pub.ext", retries=extension_manager.max_retries)
    )

    assert socket_calls == [True]


def test_find_installed_and_exclude_installed(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text(
        json.dumps([{"identifier": {"id": "pub.ext"}, "version": "1.0.0"}])
    )

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    installed = asyncio.run(bare_manager.find_installed())
    assert installed[0]["identifier"]["id"] == "pub.ext"

    bare_manager.installed = installed
    bare_manager.extensions = ["pub.ext", "other.ext"]
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0"},
        "other.ext": {"version": "9.9.9"},
    }

    asyncio.run(bare_manager.exclude_installed())

    assert bare_manager.extensions == ["other.ext"]


def test_exclude_installed_keeps_non_matching_entries(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.installed = [{"identifier": {"id": "pub.ext"}, "version": "1.0.0"}]
    bare_manager.extensions = ["different.ext"]
    bare_manager.extension_metadata = {"different.ext": {"version": "9.9.9"}}

    asyncio.run(bare_manager.exclude_installed())

    assert bare_manager.extensions == ["different.ext"]


def test_extensions_json_path_uses_vscode_root(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "vscode-root"
    monkeypatch.setattr(extension_manager, "vscode_root", root)

    assert (
        bare_manager.extensions_json_path() == root / "extensions" / "extensions.json"
    )


def test_cli_install_function_uses_expected_constructor_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Manager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            captured["config_name"] = config_name
            captured["code_path"] = code_path
            captured["target_directory"] = target_directory

        async def initialize(self) -> None:
            captured["initialized"] = True

        async def install_async(self) -> None:
            captured["installed"] = True

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _Manager)

    extension_manager.install(
        config_name="devcontainer.custom.json",
        code_path="/custom/code",
        target_path="/ignored/by/current-api",
        log_level="info",
    )

    assert captured == {
        "config_name": "devcontainer.custom.json",
        "code_path": "/custom/code",
        "target_directory": "/ignored/by/current-api",
        "initialized": True,
        "installed": True,
    }


def test_initialize_sets_extensions_and_installed(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _initialize() -> None:
        return None

    async def _parse() -> list[str]:
        return ["one", "two"]

    async def _find() -> list[dict[str, Any]]:
        return [{"identifier": {"id": "one"}}]

    bare_manager.socket_manager = cast(Any, SimpleNamespace(initialize=_initialize))
    monkeypatch.setattr(bare_manager, "parse_all_extensions", _parse)
    monkeypatch.setattr(bare_manager, "find_installed", _find)

    asyncio.run(bare_manager.initialize())

    assert bare_manager.extensions == ["one", "two"]
    assert bare_manager.installed == [{"identifier": {"id": "one"}}]


def test_install_async_iterates_and_sleeps(
    bare_manager: CodeExtensionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare_manager.extensions = ["one", "two"]
    called: list[str] = []

    async def _exclude() -> None:
        called.append("exclude")

    monkeypatch.setattr(bare_manager, "exclude_installed", _exclude)

    async def _install_extension(
        extension_id: str, extension_pack: bool = False, retries: int = 0
    ) -> None:
        called.append(f"install:{extension_id}:{extension_pack}:{retries}")

    async def _sleep(_seconds: int) -> None:
        called.append("sleep")

    @contextmanager
    def _provisioner(install_dir):
        called.append(f"provision:{install_dir}")
        yield Path("/tmp/vsce-sign")
        called.append("cleanup")

    monkeypatch.setattr(bare_manager, "install_extension", _install_extension)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        extension_manager,
        "provision_vsce_sign_binary_for_run",
        _provisioner,
    )

    asyncio.run(bare_manager.install_async())

    assert called == [
        "exclude",
        f"provision:{bare_manager.target_path}",
        "install:one:False:0",
        "sleep",
        "install:two:False:0",
        "sleep",
        "cleanup",
    ]


def test_install_extension_handles_extension_pack_and_recursion_async(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "main.ext": {"version": "1.0.0", "extension_pack": ["child.ext"]},
        "child.ext": {"version": "1.0.0", "extension_pack": []},
    }
    bare_manager.extensions = ["child.ext"]
    (bare_manager.target_path / "main.ext-1.0.0.vsix").write_text("main")
    (bare_manager.target_path / "child.ext-1.0.0.vsix").write_text("child")

    updates: list[list[str]] = []
    manual: list[str] = []

    async def _update(extension_id: str = "", extension_ids: list[str] = []) -> None:
        updates.append(list(extension_ids))

    async def _manual(
        extension_id: str, extension_path: Path, update_json: bool = True
    ) -> None:
        manual.append(extension_id)

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update)
    monkeypatch.setattr(bare_manager, "install_extension_manually", _manual)

    asyncio.run(bare_manager.install_extension("main.ext"))

    assert updates == [["main.ext", "child.ext"]]
    assert manual == ["child.ext", "main.ext"]


def test_install_extension_downloads_when_file_missing_async(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    downloaded_path = bare_manager.target_path / "downloaded.vsix"
    downloaded_path.write_text("payload")

    async def _download(extension_id: str) -> Path:
        return downloaded_path

    monkeypatch.setattr(bare_manager, "download_extension", _download)
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="ok", stderr=""),
    )

    asyncio.run(bare_manager.install_extension("pub.ext"))


def test_install_extension_retries_once_on_subprocess_error_async(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    (bare_manager.target_path / "pub.ext-1.0.0.vsix").write_text("payload")

    attempts = {"count": 0}
    socket_calls: list[bool] = []

    def _run(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise subprocess.CalledProcessError(1, ["code"], "", "")
        return SimpleNamespace(stdout="ok", stderr="")

    bare_manager.socket_manager = cast(
        Any,
        SimpleNamespace(
            find_socket=lambda update_environment=False: _find_socket_collector(
                socket_calls, update_environment
            )
        ),
    )
    monkeypatch.setattr(extension_manager.subprocess, "run", _run)

    asyncio.run(bare_manager.install_extension("pub.ext"))

    assert attempts["count"] == 2
    assert socket_calls == [True]


def test_install_extension_treats_error_text_as_failure_async(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "extension_pack": []}
    }
    (bare_manager.target_path / "pub.ext-1.0.0.vsix").write_text("payload")

    socket_calls: list[bool] = []
    bare_manager.socket_manager = cast(
        Any,
        SimpleNamespace(
            find_socket=lambda update_environment=False: _find_socket_collector(
                socket_calls, update_environment
            )
        ),
    )

    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", stderr="Error: boom"),
    )

    asyncio.run(
        bare_manager.install_extension("pub.ext", retries=extension_manager.max_retries)
    )

    assert socket_calls == [True]


def test_install_extension_pack_mode_without_pending_child_async(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.extension_metadata = {
        "main.ext": {"version": "1.0.0", "extension_pack": ["child.ext"]}
    }
    bare_manager.extensions = []
    (bare_manager.target_path / "main.ext-1.0.0.vsix").write_text("payload")

    updates: list[list[str]] = []
    manual: list[str] = []

    async def _update(extension_id: str = "", extension_ids: list[str] = []) -> None:
        updates.append(list(extension_ids))

    async def _manual(
        extension_id: str, extension_path: Path, update_json: bool = True
    ) -> None:
        manual.append(extension_id)

    monkeypatch.setattr(bare_manager, "update_extensions_json", _update)
    monkeypatch.setattr(bare_manager, "install_extension_manually", _manual)

    asyncio.run(bare_manager.install_extension("main.ext", extension_pack=True))

    assert updates == [["child.ext"]]
    assert manual == ["main.ext"]


def test_build_parser_and_main_command_path(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = extension_manager.build_parser()
    parsed = parser.parse_args(["install", "--config-name", "a.json"])
    assert parsed.command == "install"
    assert parsed.config_name == "a.json"

    called: dict[str, str] = {}

    monkeypatch.setattr(
        extension_manager,
        "install",
        lambda config_name, code_path, target_path, log_level: called.update(
            {
                "config_name": config_name,
                "code_path": code_path,
                "target_path": target_path,
                "log_level": log_level,
            }
        ),
    )
    monkeypatch.setattr(
        extension_manager.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config_name="cfg.json",
            code_path="code-bin",
            target_path="/tmp/cache",
            log_level="debug",
        ),
    )

    extension_manager.main()

    assert called == {
        "config_name": "cfg.json",
        "code_path": "code-bin",
        "target_path": "/tmp/cache",
        "log_level": "debug",
    }


def test_module_main_guard_executes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["uvscem", "--help"])
    monkeypatch.delitem(sys.modules, "uvscem.extension_manager", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("uvscem.extension_manager", run_name="__main__")

    assert exc.value.code == 0
