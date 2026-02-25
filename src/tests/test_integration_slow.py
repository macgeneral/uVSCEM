from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from uvscem.extension_manager import CodeExtensionManager
from uvscem.vsce_sign_bootstrap import provision_vsce_sign_binary_for_run

pytestmark = pytest.mark.slow


ASSETS_DIR = Path(__file__).parent / "assets"


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


@pytest.mark.slow
def test_integration_fetch_install_uninstall_with_verification(tmp_path: Path) -> None:
    code_binary = shutil.which("code")
    if code_binary is None:
        pytest.skip("code CLI is not available on PATH")

    config_path = ASSETS_DIR / "test_extensions.json"
    extensions = _load_extensions(config_path)

    already_installed = _installed_extensions(code_binary)
    if any(extension_id in already_installed for extension_id in extensions):
        pytest.skip("One or more slow-test extensions are already installed")

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=code_binary,
        target_directory=str(tmp_path / "cache"),
    )

    try:
        asyncio.run(manager.initialize())
        asyncio.run(manager.install_async())

        installed_now = _installed_extensions(code_binary)
        for extension_id in extensions:
            assert extension_id in installed_now
    finally:
        for extension_id in extensions:
            _uninstall_extension(code_binary, extension_id)

    installed_after_cleanup = _installed_extensions(code_binary)
    for extension_id in extensions:
        assert extension_id not in installed_after_cleanup


@pytest.mark.slow
def test_integration_tampered_vsix_fails_signature_verification(tmp_path: Path) -> None:
    config_path = ASSETS_DIR / "test_extensions.json"
    extension_id = "dbaeumer.vscode-eslint"

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=shutil.which("code") or "code",
        target_directory=str(tmp_path / "cache"),
    )

    asyncio.run(manager.initialize())
    metadata = manager.extension_metadata.get(extension_id, {})
    if not metadata:
        pytest.skip("Could not resolve extension metadata for slow integration test")

    vsix_path = asyncio.run(manager.download_extension(extension_id))
    asyncio.run(manager.download_signature_archive(extension_id))

    payload = bytearray(vsix_path.read_bytes())
    if not payload:
        pytest.skip("Downloaded VSIX is empty")
    payload[len(payload) // 2] ^= 0x01
    vsix_path.write_bytes(bytes(payload))

    with provision_vsce_sign_binary_for_run(install_dir=tmp_path / "bin") as binary:
        manager.vsce_sign_binary = binary
        try:
            with pytest.raises(subprocess.CalledProcessError):
                asyncio.run(manager.install_extension(extension_id))
        finally:
            manager.vsce_sign_binary = None
