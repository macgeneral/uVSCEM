from __future__ import annotations

import asyncio
import json
import runpy
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from uvscem import extension_manager
from uvscem.exceptions import (
    InstallationWorkflowError,
    OfflineBundleExportError,
    OfflineBundleImportExecutionError,
    OfflineModeError,
)
from uvscem.extension_manager import CodeExtensionManager
from uvscem.models import ExtensionSpec


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


def test_parse_extension_spec_supports_pinned_versions(
    bare_manager: CodeExtensionManager,
) -> None:
    assert bare_manager.parse_extension_spec("   ") == ("", "")
    assert bare_manager.parse_extension_spec("publisher.name") == (
        "publisher.name",
        "",
    )
    assert bare_manager.parse_extension_spec("publisher.name@") == (
        "publisher.name@",
        "",
    )
    assert bare_manager.parse_extension_spec("@1.2.3") == ("@1.2.3", "")
    assert bare_manager.parse_extension_spec("publisher.name@1.2.3") == (
        "publisher.name",
        "1.2.3",
    )
    assert bare_manager.parse_extension_spec("publisher.name@1.2.3-pre") == (
        "publisher.name",
        "1.2.3-pre",
    )


def test_resolve_extension_requests_skips_empty_specs(
    bare_manager: CodeExtensionManager,
) -> None:
    extensions, extension_pins = asyncio.run(
        bare_manager._resolve_extension_requests(
            [
                ExtensionSpec(extension_id="", pinned_version=""),
                ExtensionSpec(extension_id="publisher.name", pinned_version="1.2.3"),
            ]
        )
    )

    assert extensions == ["publisher.name"]
    assert extension_pins == {"publisher.name": "1.2.3"}


def test_init_sets_paths_and_creates_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "devcontainer.json"
    config_path.write_text("{}")
    target_dir = tmp_path / "target-cache"
    code_binary = tmp_path / "code"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(extension_manager, "CodeAPIManager", lambda: SimpleNamespace())
    monkeypatch.setattr(extension_manager, "CodeManager", lambda: SimpleNamespace())
    manager = CodeExtensionManager(
        config_name="devcontainer.json",
        code_path=str(code_binary),
        target_directory=str(target_dir),
    )

    assert manager.code_binary == str(code_binary)
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

    async def _metadata(extension_id: str, **_kwargs):
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


def test_parse_all_extensions_resolves_pinned_versions(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.dev_container_config_path.write_text("{}")

    monkeypatch.setattr(
        extension_manager.json5,
        "loads",
        lambda _value: {
            "customizations": {
                "vscode": {"extensions": ["publisher.alpha@1.9.0", "publisher.beta"]}
            }
        },
    )

    calls: list[tuple[str, bool, str]] = []

    async def _metadata(
        extension_id: str,
        include_latest_stable_version_only: bool = True,
        requested_version: str = "",
        **_kwargs,
    ):
        calls.append(
            (extension_id, include_latest_stable_version_only, requested_version)
        )
        if extension_id == "publisher.alpha":
            return {
                extension_id: [
                    {
                        "version": "1.9.0",
                        "dependencies": [],
                        "extension_pack": [],
                    }
                ]
            }
        return {
            extension_id: [
                {
                    "version": "2.0.0",
                    "dependencies": [],
                    "extension_pack": [],
                }
            ]
        }

    bare_manager.api_manager = cast(
        Any, SimpleNamespace(get_extension_metadata=_metadata)
    )

    class _Dependencies:
        def __init__(self, deps):
            self.deps = deps

        def resolve_dependencies(self):
            return ["publisher.alpha", "publisher.beta"]

    monkeypatch.setattr(extension_manager, "Dependencies", _Dependencies)

    resolved = asyncio.run(bare_manager.parse_all_extensions())

    assert resolved == ["publisher.alpha", "publisher.beta"]
    assert bare_manager.extension_metadata["publisher.alpha"]["version"] == "1.9.0"
    assert bare_manager.extension_metadata["publisher.beta"]["version"] == "2.0.0"
    assert calls == [
        ("publisher.alpha", False, "1.9.0"),
        ("publisher.beta", True, ""),
    ]


def test_parse_all_extensions_raises_when_pinned_version_missing(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare_manager.dev_container_config_path.write_text("{}")

    monkeypatch.setattr(
        extension_manager.json5,
        "loads",
        lambda _value: {
            "customizations": {"vscode": {"extensions": ["publisher.alpha@9.9.9"]}}
        },
    )

    async def _metadata(
        extension_id: str,
        include_latest_stable_version_only: bool = True,
        requested_version: str = "",
        **_kwargs,
    ):
        del extension_id, include_latest_stable_version_only, requested_version
        return {"publisher.alpha": []}

    bare_manager.api_manager = cast(
        Any, SimpleNamespace(get_extension_metadata=_metadata)
    )

    with pytest.raises(ValueError, match="Pinned extension version not found"):
        asyncio.run(bare_manager.parse_all_extensions())


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

    @contextmanager
    def _noop_provisioner(install_dir):
        yield None

    monkeypatch.setattr(bare_manager, "exclude_installed", _exclude)
    monkeypatch.setattr(bare_manager, "install_extension", _install)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        extension_manager, "provision_vsce_sign_binary_for_run", _noop_provisioner
    )

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


def test_download_extension_reuses_cached_file(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "url": "https://download/file.vsix"}
    }
    cached_path = bare_manager.target_path / bare_manager.get_filename("pub.ext")
    cached_path.write_bytes(b"cached")

    bare_manager.api_manager = cast(
        Any,
        SimpleNamespace(
            session=SimpleNamespace(
                get=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("network should not be called when cache exists")
                )
            )
        ),
    )

    downloaded = asyncio.run(bare_manager.download_extension("pub.ext"))

    assert downloaded == cached_path
    assert downloaded.read_bytes() == b"cached"


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


def test_download_signature_archive_reuses_cached_file(
    bare_manager: CodeExtensionManager,
) -> None:
    bare_manager.extension_metadata = {
        "pub.ext": {"version": "1.0.0", "signature": "https://download/file.sigzip"}
    }
    cached_path = bare_manager.target_path / bare_manager.get_signature_filename(
        "pub.ext"
    )
    cached_path.write_bytes(b"cached-signature")

    bare_manager.api_manager = cast(
        Any,
        SimpleNamespace(
            session=SimpleNamespace(
                get=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("network should not be called when cache exists")
                )
            )
        ),
    )

    downloaded = asyncio.run(bare_manager.download_signature_archive("pub.ext"))

    assert downloaded == cached_path
    assert downloaded.read_bytes() == b"cached-signature"


def test_verify_extension_signature_runs_vsce_sign(
    bare_manager: CodeExtensionManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vsix_path = tmp_path / "ext.vsix"
    sig_path = tmp_path / "ext.sigzip"
    vsix_path.write_text("payload")
    sig_path.write_text("payload")
    vsce_sign_binary = Path(tempfile.gettempdir()) / "vsce-sign"
    bare_manager.vsce_sign_binary = vsce_sign_binary
    called: dict[str, list[str]] = {}

    def _run(cmd, capture_output, check, text):
        called["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(extension_manager.subprocess, "run", _run)

    asyncio.run(bare_manager.verify_extension_signature(vsix_path, sig_path))

    assert called["cmd"][0] == str(vsce_sign_binary)


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
    bare_manager.vsce_sign_binary = Path(tempfile.gettempdir()) / "vsce-sign"

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


def test_update_extensions_json_raises_and_keeps_original_on_write_error(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text('[{"existing": true}]', encoding="utf-8")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)
    monkeypatch.setattr(
        extension_manager,
        "_write_json_atomic",
        lambda _path, _payload: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(bare_manager.update_extensions_json(extension_ids=[]))

    assert json_path.read_text(encoding="utf-8") == '[{"existing": true}]'


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


def test_update_extensions_json_default_extension_ids_do_not_leak_between_calls(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)

    bare_manager.extension_metadata = {
        "pub.ext": {"installation_metadata": {"identifier": {"id": "pub.ext"}}},
        "other.ext": {"installation_metadata": {"identifier": {"id": "other.ext"}}},
    }

    asyncio.run(bare_manager.update_extensions_json(extension_id="pub.ext"))
    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {"identifier": {"id": "pub.ext"}}
    ]

    asyncio.run(bare_manager.update_extensions_json(extension_id="other.ext"))
    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {"identifier": {"id": "other.ext"}}
    ]


def test_update_extensions_json_replaces_existing_entry_with_same_identifier(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)

    async def _find_installed() -> list[dict[str, Any]]:
        return [
            {
                "identifier": {"id": "pub.ext"},
                "version": "1.0.0",
                "metadata": {"legacy": True},
            },
            {
                "identifier": {"id": "other.ext"},
                "version": "9.9.9",
            },
        ]

    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)
    bare_manager.extension_metadata = {
        "pub.ext": {
            "installation_metadata": {
                "identifier": {"id": "pub.ext"},
                "version": "2.0.0",
                "metadata": {"legacy": False},
            }
        }
    }

    asyncio.run(bare_manager.update_extensions_json(extension_id="pub.ext"))

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == [
        {
            "identifier": {"id": "pub.ext"},
            "version": "2.0.0",
            "metadata": {"legacy": False},
        },
        {"identifier": {"id": "other.ext"}, "version": "9.9.9"},
    ]


def test_update_extensions_json_deduplicates_multiple_existing_entries(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)

    async def _find_installed() -> list[dict[str, Any]]:
        return [
            {"identifier": {"id": "pub.ext"}, "version": "1.0.0"},
            {"identifier": {"id": "pub.ext"}, "version": "1.1.0"},
        ]

    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)
    bare_manager.extension_metadata = {
        "pub.ext": {
            "installation_metadata": {
                "identifier": {"id": "pub.ext"},
                "version": "2.0.0",
            }
        }
    }

    asyncio.run(bare_manager.update_extensions_json(extension_id="pub.ext"))

    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {"identifier": {"id": "pub.ext"}, "version": "2.0.0"}
    ]


def test_update_extensions_json_appends_metadata_without_identifier(
    bare_manager: CodeExtensionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "extensions.json"
    json_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(bare_manager, "extensions_json_path", lambda: json_path)
    monkeypatch.setattr(bare_manager, "find_installed", _find_installed)
    bare_manager.extension_metadata = {
        "pub.ext": {
            "installation_metadata": {
                "version": "2.0.0",
                "metadata": {"note": "no identifier"},
            }
        }
    }

    asyncio.run(bare_manager.update_extensions_json(extension_id="pub.ext"))

    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {
            "version": "2.0.0",
            "metadata": {"note": "no identifier"},
        }
    ]


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
    bare_manager.vsce_sign_binary = Path(tempfile.gettempdir()) / "vsce-sign"

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
    bare_manager.vsce_sign_binary = Path(tempfile.gettempdir()) / "vsce-sign"

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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    code_path = tmp_path / "custom" / "code"
    target_path = tmp_path / "ignored" / "by-current-api"

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
        code_path=str(code_path),
        target_path=str(target_path),
        log_level="info",
    )

    assert captured == {
        "config_name": "devcontainer.custom.json",
        "code_path": str(code_path),
        "target_directory": str(target_path),
        "initialized": True,
        "installed": True,
    }


def test_cli_install_function_maps_errors_to_installation_workflow_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Manager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path, target_directory

        async def initialize(self) -> None:
            return None

        async def install_async(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _Manager)

    with pytest.raises(
        InstallationWorkflowError, match="Extension installation failed"
    ):
        extension_manager.install()


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
        yield Path(tempfile.gettempdir()) / "vsce-sign"
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
    cache_path = Path(tempfile.gettempdir()) / "cache"

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
            target_path=str(cache_path),
            log_level="debug",
        ),
    )

    extension_manager.main()

    assert called == {
        "config_name": "cfg.json",
        "code_path": "code-bin",
        "target_path": str(cache_path),
        "log_level": "debug",
    }


def test_build_parser_accepts_export_and_import_commands() -> None:
    parser = extension_manager.build_parser()

    parsed_export = parser.parse_args(["export"])
    parsed_import = parser.parse_args(["import"])

    assert parsed_export.command == "export"
    assert parsed_import.command == "import"


def test_main_routes_export_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}
    cache_path = Path(tempfile.gettempdir()) / "cache"
    bundle_path = Path(tempfile.gettempdir()) / "bundle"

    monkeypatch.setattr(
        extension_manager,
        "export_offline_bundle",
        lambda config_name, bundle_path, target_path, code_path, log_level, vsce_sign_version, vsce_sign_targets, manifest_signing_key: (
            called.update(
                {
                    "config_name": config_name,
                    "bundle_path": bundle_path,
                    "target_path": target_path,
                    "code_path": code_path,
                    "log_level": log_level,
                    "vsce_sign_version": vsce_sign_version,
                    "vsce_sign_targets": vsce_sign_targets,
                    "manifest_signing_key": manifest_signing_key,
                }
            )
        ),
    )
    monkeypatch.setattr(
        extension_manager.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            command="export",
            config_name="cfg.json",
            bundle_path=str(bundle_path),
            target_path=str(cache_path),
            code_path="code-bin",
            log_level="debug",
            vsce_sign_version="2.0.6",
            vsce_sign_targets="current",
            manifest_signing_key="ABC123",
        ),
    )

    extension_manager.main()

    assert called == {
        "config_name": "cfg.json",
        "bundle_path": str(bundle_path),
        "target_path": str(cache_path),
        "code_path": "code-bin",
        "log_level": "debug",
        "vsce_sign_version": "2.0.6",
        "vsce_sign_targets": "current",
        "manifest_signing_key": "ABC123",
    }


def test_main_routes_import_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}
    cache_path = Path(tempfile.gettempdir()) / "cache"
    bundle_path = Path(tempfile.gettempdir()) / "bundle"
    keyring_path = Path(tempfile.gettempdir()) / "keyring.gpg"

    monkeypatch.setattr(
        extension_manager,
        "import_offline_bundle",
        lambda bundle_path, code_path, target_path, log_level, strict_offline, verify_manifest_signature, manifest_verification_keyring: (
            called.update(
                {
                    "bundle_path": bundle_path,
                    "code_path": code_path,
                    "target_path": target_path,
                    "log_level": log_level,
                    "strict_offline": str(strict_offline),
                    "verify_manifest_signature": str(verify_manifest_signature),
                    "manifest_verification_keyring": manifest_verification_keyring,
                }
            )
        ),
    )
    monkeypatch.setattr(
        extension_manager.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            command="import",
            config_name="cfg.json",
            bundle_path=str(bundle_path),
            target_path=str(cache_path),
            code_path="code-bin",
            log_level="debug",
            vsce_sign_version="2.0.6",
            strict_offline=True,
            verify_manifest_signature=True,
            manifest_verification_keyring=str(keyring_path),
        ),
    )

    extension_manager.main()

    assert called == {
        "bundle_path": str(bundle_path),
        "code_path": "code-bin",
        "target_path": str(cache_path),
        "log_level": "debug",
        "strict_offline": "True",
        "verify_manifest_signature": "True",
        "manifest_verification_keyring": str(keyring_path),
    }


def test_run_command_returns_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda cmd, capture_output, check, text: SimpleNamespace(
            returncode=0,
            stdout="ok",
            stderr="",
        ),
    )

    extension_manager._run_command(["echo", "ok"])


def test_run_command_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extension_manager.subprocess,
        "run",
        lambda cmd, capture_output, check, text: SimpleNamespace(
            returncode=2,
            stdout="oops",
            stderr="bad",
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        extension_manager._run_command(["false"])


def test_sign_bundle_manifest_invokes_gpg_and_returns_signature_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    called: list[list[str]] = []

    def _run_command(cmd: list[str]) -> None:
        called.append(cmd)

    monkeypatch.setattr(extension_manager, "_run_command", _run_command)

    signature_path = extension_manager._sign_bundle_manifest(manifest_path, "ABC123")

    assert signature_path == manifest_path.with_suffix(".json.asc")
    assert called == [
        [
            "gpg",
            "--batch",
            "--yes",
            "--armor",
            "--local-user",
            "ABC123",
            "--output",
            str(manifest_path.with_suffix(".json.asc")),
            "--detach-sign",
            str(manifest_path),
        ]
    ]


def test_verify_bundle_manifest_signature_invokes_gpg_without_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    signature_path = tmp_path / "manifest.json.asc"
    called: list[list[str]] = []

    def _run_command(cmd: list[str]) -> None:
        called.append(cmd)

    monkeypatch.setattr(extension_manager, "_run_command", _run_command)

    extension_manager._verify_bundle_manifest_signature(manifest_path, signature_path)

    assert called == [
        [
            "gpg",
            "--batch",
            "--verify",
            str(signature_path),
            str(manifest_path),
        ]
    ]


def test_verify_bundle_manifest_signature_invokes_gpg_with_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    signature_path = tmp_path / "manifest.json.asc"
    keyring_path = tmp_path / "keyring.gpg"
    called: list[list[str]] = []

    def _run_command(cmd: list[str]) -> None:
        called.append(cmd)

    monkeypatch.setattr(extension_manager, "_run_command", _run_command)

    extension_manager._verify_bundle_manifest_signature(
        manifest_path,
        signature_path,
        verification_keyring=str(keyring_path),
    )

    assert called == [
        [
            "gpg",
            "--batch",
            "--no-default-keyring",
            "--keyring",
            str(keyring_path),
            "--verify",
            str(signature_path),
            str(manifest_path),
        ]
    ]


def test_export_offline_bundle_writes_manifest_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"

    class _ExportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            self.config_name = config_name
            self.code_path = code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {
                "publisher.demo": {
                    "version": "1.2.3",
                    "installation_metadata": {"identifier": {"id": "publisher.demo"}},
                    "dependencies": ["publisher.dep"],
                    "extension_pack": ["publisher.pack"],
                }
            }

        async def parse_all_extensions(self) -> list[str]:
            return ["publisher.demo"]

        async def download_extension(self, extension_id: str) -> Path:
            file_path = self.target_path / f"{extension_id}-1.2.3.vsix"
            file_path.write_text("vsix")
            return file_path

        async def download_signature_archive(self, extension_id: str) -> Path:
            file_path = self.target_path / f"{extension_id}-1.2.3.sigzip"
            file_path.write_text("sig")
            return file_path

    def _install_vsce_sign_binary_for_target(
        target: str,
        install_dir: str | Path,
        version: str,
        force: bool = False,
        session: Any | None = None,
        verify_existing_checksum: bool = True,
    ) -> Path:
        del version, force, session, verify_existing_checksum
        install_path = Path(install_dir)
        install_path.mkdir(parents=True, exist_ok=True)
        binary_name = "vsce-sign.exe" if target.startswith("win32-") else "vsce-sign"
        binary_path = install_path / binary_name
        binary_path.write_text("binary")
        return binary_path

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ExportManager)
    monkeypatch.setattr(
        extension_manager,
        "install_vsce_sign_binary_for_target",
        _install_vsce_sign_binary_for_target,
    )
    monkeypatch.setattr(
        extension_manager,
        "get_vsce_sign_package_name",
        lambda target=None: f"@vscode/vsce-sign-{target or 'linux-x64'}",
    )
    monkeypatch.setattr(
        extension_manager,
        "get_vsce_sign_target",
        lambda: "linux-x64",
    )

    extension_manager.export_offline_bundle(
        config_name="devcontainer.json",
        bundle_path=str(bundle_path),
        target_path=str(tmp_path / "cache"),
        code_path="code",
        log_level="info",
        vsce_sign_version="2.0.6",
        vsce_sign_targets="current",
    )

    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["ordered_extensions"] == ["publisher.demo"]
    assert manifest["extensions"][0]["filename"] == "publisher.demo-1.2.3.vsix"
    assert manifest["vsce_sign"]["binaries"][0]["target"] == "linux-x64"
    assert (bundle_path / "artifacts" / "publisher.demo-1.2.3.vsix").is_file()
    assert (bundle_path / "artifacts" / "publisher.demo-1.2.3.sigzip").is_file()
    assert (bundle_path / "vsce-sign" / "linux-x64" / "vsce-sign").is_file()


def test_export_offline_bundle_signs_manifest_when_key_is_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    signed: list[tuple[str, str]] = []

    class _ExportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {
                "publisher.demo": {
                    "version": "1.2.3",
                    "installation_metadata": {},
                    "dependencies": [],
                    "extension_pack": [],
                }
            }

        async def parse_all_extensions(self) -> list[str]:
            return ["publisher.demo"]

        async def download_extension(self, extension_id: str) -> Path:
            path = self.target_path / f"{extension_id}-1.2.3.vsix"
            path.write_text("vsix")
            return path

        async def download_signature_archive(self, extension_id: str) -> Path:
            path = self.target_path / f"{extension_id}-1.2.3.sigzip"
            path.write_text("sig")
            return path

    def _install_vsce_sign_binary_for_target(
        target: str,
        install_dir: str | Path,
        version: str,
        force: bool = False,
        session: Any | None = None,
        verify_existing_checksum: bool = True,
    ) -> Path:
        del version, force, session, verify_existing_checksum
        install_path = Path(install_dir)
        install_path.mkdir(parents=True, exist_ok=True)
        binary = install_path / (
            "vsce-sign.exe" if target.startswith("win32-") else "vsce-sign"
        )
        binary.write_text(target)
        return binary

    def _sign_manifest(manifest_path: Path, signing_key: str) -> Path:
        signature_path = manifest_path.with_suffix(f"{manifest_path.suffix}.asc")
        signature_path.write_text("signed")
        signed.append((str(manifest_path), signing_key))
        return signature_path

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ExportManager)
    monkeypatch.setattr(
        extension_manager,
        "install_vsce_sign_binary_for_target",
        _install_vsce_sign_binary_for_target,
    )
    monkeypatch.setattr(extension_manager, "_sign_bundle_manifest", _sign_manifest)

    extension_manager.export_offline_bundle(
        bundle_path=str(bundle_path),
        manifest_signing_key="ABC123",
    )

    assert signed == [(str(bundle_path / "manifest.json"), "ABC123")]
    assert (bundle_path / "manifest.json.asc").is_file()


def test_export_offline_bundle_supports_all_vsce_sign_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    seen_targets: list[str] = []

    class _ExportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {
                "publisher.demo": {
                    "version": "1.2.3",
                    "installation_metadata": {},
                    "dependencies": [],
                    "extension_pack": [],
                }
            }

        async def parse_all_extensions(self) -> list[str]:
            return ["publisher.demo"]

        async def download_extension(self, extension_id: str) -> Path:
            file_path = self.target_path / f"{extension_id}-1.2.3.vsix"
            file_path.write_text("vsix")
            return file_path

        async def download_signature_archive(self, extension_id: str) -> Path:
            file_path = self.target_path / f"{extension_id}-1.2.3.sigzip"
            file_path.write_text("sig")
            return file_path

    def _install_vsce_sign_binary_for_target(
        target: str,
        install_dir: str | Path,
        version: str,
        force: bool = False,
        session: Any | None = None,
        verify_existing_checksum: bool = True,
    ) -> Path:
        del version, force, session, verify_existing_checksum
        seen_targets.append(target)
        install_path = Path(install_dir)
        install_path.mkdir(parents=True, exist_ok=True)
        binary = install_path / (
            "vsce-sign.exe" if target.startswith("win32-") else "vsce-sign"
        )
        binary.write_text(target)
        return binary

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ExportManager)
    monkeypatch.setattr(
        extension_manager,
        "install_vsce_sign_binary_for_target",
        _install_vsce_sign_binary_for_target,
    )
    monkeypatch.setattr(
        extension_manager,
        "SUPPORTED_VSCE_SIGN_TARGETS",
        ("linux-x64", "win32-x64"),
    )

    extension_manager.export_offline_bundle(
        bundle_path=str(bundle_path),
        vsce_sign_targets="all",
    )

    assert seen_targets == ["linux-x64", "win32-x64"]


def test_export_offline_bundle_supports_custom_target_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    seen_targets: list[str] = []

    class _ExportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {
                "publisher.demo": {
                    "version": "1.2.3",
                    "installation_metadata": {},
                    "dependencies": [],
                    "extension_pack": [],
                }
            }

        async def parse_all_extensions(self) -> list[str]:
            return ["publisher.demo"]

        async def download_extension(self, extension_id: str) -> Path:
            path = self.target_path / f"{extension_id}-1.2.3.vsix"
            path.write_text("vsix")
            return path

        async def download_signature_archive(self, extension_id: str) -> Path:
            path = self.target_path / f"{extension_id}-1.2.3.sigzip"
            path.write_text("sig")
            return path

    def _install_vsce_sign_binary_for_target(
        target: str,
        install_dir: str | Path,
        version: str,
        force: bool = False,
        session: Any | None = None,
        verify_existing_checksum: bool = True,
    ) -> Path:
        del version, force, session, verify_existing_checksum
        seen_targets.append(target)
        install_path = Path(install_dir)
        install_path.mkdir(parents=True, exist_ok=True)
        binary = install_path / (
            "vsce-sign.exe" if target.startswith("win32-") else "vsce-sign"
        )
        binary.write_text(target)
        return binary

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ExportManager)
    monkeypatch.setattr(
        extension_manager,
        "install_vsce_sign_binary_for_target",
        _install_vsce_sign_binary_for_target,
    )

    extension_manager.export_offline_bundle(
        bundle_path=str(bundle_path),
        vsce_sign_targets="linux-x64,win32-x64",
    )

    assert seen_targets == ["linux-x64", "win32-x64"]


def test_export_offline_bundle_rejects_empty_target_list(
    tmp_path: Path,
) -> None:
    with pytest.raises(OfflineBundleExportError, match="No valid vsce-sign targets"):
        extension_manager.export_offline_bundle(
            bundle_path=str(tmp_path / "bundle"),
            vsce_sign_targets=" , ",
        )


def test_import_offline_bundle_installs_extensions_from_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    manifest = {
        "schema_version": 1,
        "ordered_extensions": ["publisher.demo"],
        "extensions": [
            {
                "id": "publisher.demo",
                "version": "1.2.3",
                "filename": "publisher.demo-1.2.3.vsix",
                "signature_filename": "publisher.demo-1.2.3.sigzip",
                "installation_metadata": {"identifier": {"id": "publisher.demo"}},
                "dependencies": [],
                "extension_pack": [],
            }
        ],
        "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
    }
    (bundle_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    installed: list[str] = []
    captured_manager: dict[str, Any] = {}

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None
            captured_manager["manager"] = self

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_pack, retries
            assert self.vsce_sign_binary is not None
            installed.append(extension_id)

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    extension_manager.import_offline_bundle(
        bundle_path=str(bundle_path),
        code_path="code",
        target_path=str(tmp_path / "cache"),
        log_level="info",
    )

    assert installed == ["publisher.demo"]
    manager = captured_manager["manager"]
    assert "publisher.demo" in manager.extension_metadata
    assert (manager.target_path / "publisher.demo-1.2.3.vsix").is_file()
    assert (manager.target_path / "publisher.demo-1.2.3.sigzip").is_file()


def test_import_offline_bundle_strict_offline_blocks_network_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "linux-x64",
                            "binary": "vsce-sign/vsce-sign",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None
            self.api_manager = SimpleNamespace(
                session=SimpleNamespace(request=lambda *_args, **_kwargs: None)
            )

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            self.api_manager.session.request("GET", "https://example.com")

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    with pytest.raises(OfflineModeError, match="strict offline mode"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
            strict_offline=True,
        )


def test_import_offline_bundle_strict_offline_handles_missing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "linux-x64",
                            "binary": "vsce-sign/vsce-sign",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None
            self.api_manager = SimpleNamespace(session=None)

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            return None

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    extension_manager.import_offline_bundle(
        bundle_path=str(bundle_path),
        target_path=str(tmp_path / "cache"),
        strict_offline=True,
    )


def test_import_offline_bundle_maps_execution_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            raise OSError("io")

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    with pytest.raises(
        OfflineBundleImportExecutionError, match="Offline bundle import failed"
    ):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_maps_runtimeerror_without_offline_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            raise RuntimeError("boom")

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    with pytest.raises(
        OfflineBundleImportExecutionError,
        match="Offline bundle import failed",
    ):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extension_manager.import_offline_bundle(bundle_path=str(tmp_path / "missing"))


def test_import_offline_bundle_raises_when_manifest_signature_missing(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir(parents=True)
    (bundle_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "ordered_extensions": [], "extensions": []}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="manifest signature"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            verify_manifest_signature=True,
        )


def test_import_offline_bundle_verifies_manifest_signature_with_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "linux-x64",
                            "binary": "vsce-sign/vsce-sign",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_path / "manifest.json.asc").write_text("signature", encoding="utf-8")

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            return None

    verified: list[tuple[str, str, str]] = []

    def _verify_manifest(
        manifest_path: Path,
        signature_path: Path,
        verification_keyring: str = "",
    ) -> None:
        verified.append((str(manifest_path), str(signature_path), verification_keyring))

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")
    monkeypatch.setattr(
        extension_manager,
        "_verify_bundle_manifest_signature",
        _verify_manifest,
    )
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    keyring_path = tmp_path / "keyring.gpg"

    extension_manager.import_offline_bundle(
        bundle_path=str(bundle_path),
        target_path=str(tmp_path / "cache"),
        verify_manifest_signature=True,
        manifest_verification_keyring=str(keyring_path),
    )

    assert verified == [
        (
            str(bundle_path / "manifest.json"),
            str(bundle_path / "manifest.json.asc"),
            str(keyring_path),
        )
    ]


def test_import_offline_bundle_raises_for_unsupported_schema(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir(parents=True)
    (bundle_path / "manifest.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported offline bundle schema version"):
        extension_manager.import_offline_bundle(bundle_path=str(bundle_path))


def test_import_offline_bundle_raises_when_extension_entry_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)

    with pytest.raises(ValueError, match="Missing extension manifest entry"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_artifacts_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    (bundle_path / "artifacts").mkdir(parents=True)
    (bundle_path / "vsce-sign").mkdir(parents=True)
    (bundle_path / "vsce-sign" / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)

    with pytest.raises(FileNotFoundError, match="Missing offline artifact"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_artifact_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                        "vsix_sha256": "deadbeef",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)

    with pytest.raises(ValueError, match="VSIX checksum mismatch"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_signature_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                        "signature_sha256": "badcafe",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)

    with pytest.raises(ValueError, match="Signature checksum mismatch"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_no_matching_vsce_sign_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    (bundle_path / "vsce-sign" / "darwin-x64").mkdir(parents=True)
    (bundle_path / "vsce-sign" / "win32-x64").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (bundle_path / "vsce-sign" / "darwin-x64" / "vsce-sign").write_text("binary")
    (bundle_path / "vsce-sign" / "win32-x64" / "vsce-sign.exe").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "darwin-x64",
                            "binary": "vsce-sign/darwin-x64/vsce-sign",
                        },
                        {
                            "target": "win32-x64",
                            "binary": "vsce-sign/win32-x64/vsce-sign.exe",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")

    with pytest.raises(ValueError, match="No bundled vsce-sign binary found"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_accepts_single_nonmatching_vsce_sign_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    (bundle_path / "vsce-sign" / "darwin-x64").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (bundle_path / "vsce-sign" / "darwin-x64" / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "darwin-x64",
                            "binary": "vsce-sign/darwin-x64/vsce-sign",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            return None

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    extension_manager.import_offline_bundle(
        bundle_path=str(bundle_path),
        target_path=str(tmp_path / "cache"),
    )


def test_import_offline_bundle_raises_when_vsce_sign_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    (bundle_path / "vsce-sign" / "linux-x64").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (bundle_path / "vsce-sign" / "linux-x64" / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {
                    "binaries": [
                        {
                            "target": "linux-x64",
                            "binary": "vsce-sign/linux-x64/vsce-sign",
                            "sha256": "bad",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager, "get_vsce_sign_target", lambda: "linux-x64")

    with pytest.raises(ValueError, match="vsce-sign checksum mismatch"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_raises_when_vsce_sign_binary_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata = {}

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)

    with pytest.raises(FileNotFoundError, match="Bundled vsce-sign binary not found"):
        extension_manager.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "cache"),
        )


def test_import_offline_bundle_keeps_existing_extensions_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "bundle"
    artifacts = bundle_path / "artifacts"
    vsce_sign_dir = bundle_path / "vsce-sign"
    artifacts.mkdir(parents=True)
    vsce_sign_dir.mkdir(parents=True)

    (artifacts / "publisher.demo-1.2.3.vsix").write_text("vsix")
    (artifacts / "publisher.demo-1.2.3.sigzip").write_text("sig")
    (vsce_sign_dir / "vsce-sign").write_text("binary")

    (bundle_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ordered_extensions": ["publisher.demo"],
                "extensions": [
                    {
                        "id": "publisher.demo",
                        "version": "1.2.3",
                        "filename": "publisher.demo-1.2.3.vsix",
                        "signature_filename": "publisher.demo-1.2.3.sigzip",
                        "installation_metadata": {
                            "identifier": {"id": "publisher.demo"}
                        },
                        "dependencies": [],
                        "extension_pack": [],
                    }
                ],
                "vsce_sign": {"binary": "vsce-sign/vsce-sign"},
            }
        ),
        encoding="utf-8",
    )

    class _ImportManager:
        def __init__(
            self, config_name: str, code_path: str, target_directory: str = ""
        ) -> None:
            del config_name, code_path
            self.target_path = Path(target_directory)
            self.target_path.mkdir(parents=True, exist_ok=True)
            self.extension_metadata: dict[str, dict[str, Any]] = {}
            self.extensions: list[str] = []
            self.installed: list[dict[str, Any]] = []
            self.vsce_sign_binary: Path | None = None

        def extensions_json_path(self) -> Path:
            return tmp_path / "vscode-root" / "extensions" / "extensions.json"

        async def find_installed(self) -> list[dict[str, Any]]:
            return []

        async def exclude_installed(self) -> None:
            return None

        async def install_extension(
            self, extension_id: str, extension_pack: bool = False, retries: int = 0
        ) -> None:
            del extension_id, extension_pack, retries
            return None

    async def _no_sleep(_seconds: int) -> None:
        return None

    monkeypatch.setattr(extension_manager, "CodeExtensionManager", _ImportManager)
    monkeypatch.setattr(extension_manager.asyncio, "sleep", _no_sleep)

    existing_json = tmp_path / "vscode-root" / "extensions" / "extensions.json"
    existing_json.parent.mkdir(parents=True, exist_ok=True)
    existing_json.write_text('[{"identifier": {"id": "existing"}}]', encoding="utf-8")

    extension_manager.import_offline_bundle(
        bundle_path=str(bundle_path),
        code_path="code",
        target_path=str(tmp_path / "cache"),
        log_level="info",
    )

    assert (
        existing_json.read_text(encoding="utf-8")
        == '[{"identifier": {"id": "existing"}}]'
    )


def test_module_main_guard_executes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["uvscem", "--help"])
    monkeypatch.delitem(sys.modules, "uvscem.extension_manager", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("uvscem.extension_manager", run_name="__main__")

    assert exc.value.code == 0
