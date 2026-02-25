from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

import uvscem.code_manager as code_manager_module
import uvscem.extension_manager as extension_manager_module
from uvscem.extension_manager import CodeExtensionManager
from uvscem.vsce_sign_bootstrap import provision_vsce_sign_binary_for_run

pytestmark = pytest.mark.slow


ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(autouse=True)
def _mock_vscode_socket_paths_for_ci(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return

    fake_socket_path = tmp_path_factory.mktemp("vscode-ipc") / "vscode-ipc-ci.sock"

    async def _fake_initialize(self) -> None:
        self.socket_path = fake_socket_path
        monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", str(fake_socket_path))

    async def _fake_find_socket(self, update_environment: bool = False) -> None:
        self.socket_path = fake_socket_path
        if update_environment:
            monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", str(fake_socket_path))

    monkeypatch.setattr(code_manager_module.CodeManager, "initialize", _fake_initialize)
    monkeypatch.setattr(
        code_manager_module.CodeManager, "find_socket", _fake_find_socket
    )


def _load_extensions(config_path: Path) -> list[str]:
    data = json.loads(config_path.read_text())
    return data.get("customizations", {}).get("vscode", {}).get("extensions", [])


def _installed_extensions(code_binary: str) -> set[str]:
    result = subprocess.run(
        [code_binary, "--list-extensions"],
        capture_output=True,
        check=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _uninstall_extension(code_binary: str, extension_id: str) -> None:
    subprocess.run(
        [code_binary, "--uninstall-extension", extension_id],
        capture_output=True,
        check=False,
        text=True,
    )


def _isolated_code_binary(real_code_binary: str, tmp_path: Path) -> str:
    extensions_dir = tmp_path / "extensions-dir"
    user_data_dir = tmp_path / "user-data-dir"
    extensions_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    wrapper = tmp_path / "code-isolated"
    wrapper.write_text(
        "#!/usr/bin/env sh\n"
        f"exec {shlex.quote(real_code_binary)} "
        f"--extensions-dir {shlex.quote(str(extensions_dir))} "
        f'--user-data-dir {shlex.quote(str(user_data_dir))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


@pytest.mark.slow
def test_integration_fetch_install_uninstall_with_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    code_binary = shutil.which("code") or "code"
    isolated_code_binary = _isolated_code_binary(code_binary, tmp_path)
    sandbox_vscode_root = tmp_path / "vscode-server"
    sandbox_extensions = sandbox_vscode_root / "extensions"
    sandbox_extensions.mkdir(parents=True, exist_ok=True)
    (sandbox_extensions / "extensions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(extension_manager_module, "vscode_root", sandbox_vscode_root)

    config_path = ASSETS_DIR / "test_extensions.json"

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=isolated_code_binary,
        target_directory=str(tmp_path / "cache"),
    )
    asyncio.run(manager.initialize())
    extensions = manager.extensions or _load_extensions(config_path)

    try:
        asyncio.run(manager.install_async())

        for extension_id in extensions:
            extension_dir = sandbox_extensions / manager.get_dirname(extension_id)
            assert extension_dir.is_dir()

        installed_metadata = json.loads(
            (sandbox_extensions / "extensions.json").read_text(encoding="utf-8")
        )
        installed_ids = {
            str(entry.get("identifier", {}).get("id", ""))
            for entry in installed_metadata
            if isinstance(entry, dict)
        }
        for extension_id in extensions:
            assert extension_id in installed_ids
    finally:
        for extension_id in extensions:
            extension_dir = sandbox_extensions / manager.get_dirname(extension_id)
            if extension_dir.is_dir():
                shutil.rmtree(extension_dir)
        (sandbox_extensions / "extensions.json").write_text("[]", encoding="utf-8")

    installed_after_cleanup = json.loads(
        (sandbox_extensions / "extensions.json").read_text(encoding="utf-8")
    )
    installed_after_cleanup_ids = {
        str(entry.get("identifier", {}).get("id", ""))
        for entry in installed_after_cleanup
        if isinstance(entry, dict)
    }
    for extension_id in extensions:
        assert extension_id not in installed_after_cleanup_ids
        extension_dir = sandbox_extensions / manager.get_dirname(extension_id)
        assert not extension_dir.exists()


@pytest.mark.slow
def test_integration_tampered_vsix_fails_signature_verification(tmp_path: Path) -> None:
    config_path = ASSETS_DIR / "test_extensions.json"
    extension_id = "dbaeumer.vscode-eslint"

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=shutil.which("code") or "code",
        target_directory=str(tmp_path / "cache"),
    )

    metadata_map = asyncio.run(manager.api_manager.get_extension_metadata(extension_id))
    versions = metadata_map.get(extension_id, [])
    if not versions:
        pytest.skip("Could not resolve extension metadata for slow integration test")
    manager.extension_metadata[extension_id] = versions[0]

    vsix_path = asyncio.run(manager.download_extension(extension_id))
    signature_path = asyncio.run(manager.download_signature_archive(extension_id))

    payload = bytearray(vsix_path.read_bytes())
    if not payload:
        pytest.skip("Downloaded VSIX is empty")
    payload[len(payload) // 2] ^= 0x01
    vsix_path.write_bytes(bytes(payload))

    with provision_vsce_sign_binary_for_run(install_dir=tmp_path / "bin") as binary:
        manager.vsce_sign_binary = binary
        try:
            with pytest.raises(subprocess.CalledProcessError):
                asyncio.run(
                    manager.verify_extension_signature(vsix_path, signature_path)
                )
        finally:
            manager.vsce_sign_binary = None
