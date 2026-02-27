from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

import uvscem.code_manager as code_manager_module
import uvscem.extension_manager as extension_manager_module
from uvscem.extension_manager import CodeExtensionManager
from uvscem.vsce_sign_bootstrap import provision_vsce_sign_binary_for_run

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("error")]


ASSETS_DIR = Path(__file__).parent / "assets"


def _artifact_cache_dir(tmp_path: Path, name: str) -> Path:
    cache_root = os.environ.get("UVSCEM_INTEGRATION_ARTIFACT_CACHE", "").strip()
    if cache_root:
        base_path = Path(cache_root).expanduser()
    else:
        project_root = Path(__file__).resolve().parents[2]
        base_path = project_root / ".tmp" / "uvscem-test-artifacts"

    try:
        base_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        base_path = Path.home() / ".cache" / "uvscem-test-artifacts"
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            base_path = tmp_path / "cache"
    target_path = base_path / name
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def _clear_cached_extension_artifacts(tmp_path: Path, extension_id: str) -> None:
    cache_dir = _artifact_cache_dir(tmp_path, "downloads")
    for file_path in cache_dir.glob(f"{extension_id}-*.vsix"):
        file_path.unlink(missing_ok=True)
    for file_path in cache_dir.glob(f"{extension_id}-*.sigzip"):
        file_path.unlink(missing_ok=True)


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


@contextmanager
def _misconfigured_proxy(monkeypatch: pytest.MonkeyPatch):
    proxy_url = "http://127.0.0.1:9"
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.setenv("https_proxy", proxy_url)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    yield


@pytest.mark.slow
def test_integration_fetch_install_uninstall_with_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    code_binary = shutil.which("code")
    isolated_code_binary = (
        _isolated_code_binary(code_binary, tmp_path) if code_binary else "code"
    )
    sandbox_vscode_root = tmp_path / "vscode-server"
    sandbox_extensions = sandbox_vscode_root / "extensions"
    sandbox_extensions.mkdir(parents=True, exist_ok=True)
    (sandbox_extensions / "extensions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(extension_manager_module, "vscode_root", sandbox_vscode_root)

    config_path = ASSETS_DIR / "test_extensions.json"

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=isolated_code_binary,
        target_directory=str(_artifact_cache_dir(tmp_path, "downloads")),
    )
    asyncio.run(manager.initialize())
    extensions = manager.extensions or _load_extensions(config_path)

    try:
        if not code_binary:
            pytest.skip("VS Code CLI not available for install/uninstall validation")
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
        target_directory=str(_artifact_cache_dir(tmp_path, "downloads")),
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
    tampered_vsix_path = tmp_path / "tampered.vsix"
    tampered_vsix_path.write_bytes(bytes(payload))

    with provision_vsce_sign_binary_for_run(
        install_dir=_artifact_cache_dir(tmp_path, "vsce-sign-bin"),
        verify_existing_checksum=False,
    ) as binary:
        manager.vsce_sign_binary = binary
        try:
            with pytest.raises(subprocess.CalledProcessError):
                asyncio.run(
                    manager.verify_extension_signature(
                        tampered_vsix_path, signature_path
                    )
                )
        finally:
            manager.vsce_sign_binary = None


@pytest.mark.slow
def test_integration_http_store_access_with_allow_http_uses_initial_http_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ASSETS_DIR / "test_extensions.json"
    extension_id = "dbaeumer.vscode-eslint"
    _clear_cached_extension_artifacts(tmp_path, extension_id)

    metadata_manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=shutil.which("code") or "code",
        target_directory=str(_artifact_cache_dir(tmp_path, "downloads-http")),
    )
    metadata_map = asyncio.run(
        metadata_manager.api_manager.get_extension_metadata(extension_id)
    )
    versions = metadata_map.get(extension_id, [])
    if not versions:
        pytest.skip("Could not resolve extension metadata for slow integration test")

    manager = CodeExtensionManager(
        config_name=str(config_path),
        code_path=shutil.which("code") or "code",
        target_directory=str(_artifact_cache_dir(tmp_path, "downloads-http")),
        allow_http=True,
    )

    metadata = dict(versions[0])
    original_url = str(metadata.get("url", ""))
    if not original_url.startswith("https://"):
        pytest.skip(
            "Extension URL did not resolve to HTTPS, cannot build HTTP test URL"
        )

    metadata["url"] = original_url.replace("https://", "http://", 1)
    manager.extension_metadata[extension_id] = metadata

    requested_urls: list[str] = []

    def _recording_get(url: str, *args, **kwargs):
        del args, kwargs
        requested_urls.append(str(url))
        raise requests.RequestException("stop after initial request capture")

    monkeypatch.setattr(manager.api_manager.session, "get", _recording_get)

    try:
        vsix_path = asyncio.run(manager.download_extension(extension_id))
        assert vsix_path.is_file()
        assert vsix_path.stat().st_size > 0
    except requests.RequestException:
        pass

    assert requested_urls
    assert requested_urls[0].startswith("http://")


@pytest.mark.slow
@pytest.mark.parametrize(
    "extension_id",
    ["dbaeumer.vscode-eslint", "cristianvasquez1312.hadar-vscode"],
    ids=["extension", "extension-pack"],
)
@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "pinned-older"])
def test_integration_offline_bundle_import_ci_compatible_with_misconfigured_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extension_id: str,
    pinned: bool,
) -> None:
    _clear_cached_extension_artifacts(tmp_path, extension_id)

    if pinned:
        metadata_map = asyncio.run(
            CodeExtensionManager(
                config_name=str(ASSETS_DIR / "test_extensions.json"),
                code_path=shutil.which("code") or "code",
                target_directory=str(_artifact_cache_dir(tmp_path, "downloads")),
            ).api_manager.get_extension_metadata(
                extension_id,
                include_latest_stable_version_only=False,
            )
        )
        versions = metadata_map.get(extension_id, [])
        latest_version = str(versions[0].get("version", "")) if versions else ""
        pinned_metadata = next(
            (
                version
                for version in versions[1:]
                if str(version.get("version", ""))
                and str(version.get("signature", ""))
                and str(version.get("version", "")) != latest_version
            ),
            None,
        )
        if pinned_metadata is None:
            pinned_metadata = next(
                (
                    version
                    for version in versions
                    if str(version.get("version", ""))
                    and str(version.get("signature", ""))
                ),
                None,
            )
        if pinned_metadata is None:
            pytest.skip("Could not resolve older extension version for pinned test")
        resolved_version = str(pinned_metadata.get("version", ""))
        if not resolved_version:
            pytest.skip("Could not resolve extension version for pinned test")
        extension_spec = f"{extension_id}@{resolved_version}"
    else:
        resolved_version = ""
        extension_spec = extension_id

    config_path = tmp_path / (
        "test_extensions_pinned.json" if pinned else "test_extensions_unpinned.json"
    )
    config_path.write_text(
        json.dumps(
            {
                "customizations": {
                    "vscode": {
                        "extensions": [extension_spec],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    bundle_path = tmp_path / "bundle"
    code_binary = shutil.which("code")
    sandbox_vscode_root = tmp_path / "vscode-server"
    sandbox_extensions = sandbox_vscode_root / "extensions"
    sandbox_extensions.mkdir(parents=True, exist_ok=True)
    (sandbox_extensions / "extensions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(extension_manager_module, "vscode_root", sandbox_vscode_root)
    ordered_extensions: list[str] = []
    try:
        extension_manager_module.export_offline_bundle(
            config_name=str(config_path),
            bundle_path=str(bundle_path),
            target_path=str(_artifact_cache_dir(tmp_path, "downloads")),
            code_path=code_binary or "code",
        )

        manifest = json.loads(
            (bundle_path / "manifest.json").read_text(encoding="utf-8")
        )
        ordered_extensions = [
            str(resolved_extension_id)
            for resolved_extension_id in manifest.get("ordered_extensions", [])
        ]
        extension_entries = {
            str(entry.get("id", "")): entry for entry in manifest.get("extensions", [])
        }
        assert extension_id in set(ordered_extensions)

        if pinned:
            assert str(extension_entries.get(extension_id, {}).get("version", "")) == (
                resolved_version
            )

        if not code_binary:
            return

        with _misconfigured_proxy(monkeypatch):
            extension_manager_module.import_offline_bundle(
                bundle_path=str(bundle_path),
                target_path=str(tmp_path / "import-cache"),
                code_path=code_binary,
                strict_offline=True,
                verify_manifest_signature=False,
            )

        assert extension_id in set(ordered_extensions)
    finally:
        if code_binary:
            for installed_extension_id in {*ordered_extensions, extension_id}:
                _uninstall_extension(code_binary, installed_extension_id)


@pytest.mark.slow
def test_integration_offline_bundle_import_local_vscode_install_with_misconfigured_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        pytest.skip("Local VS Code CLI install validation is skipped on CI")

    code_binary = shutil.which("code")
    if not code_binary:
        pytest.skip("VS Code CLI not available")

    extension_id = "dbaeumer.vscode-eslint"
    config_path = tmp_path / "single-extension.json"
    config_path.write_text(
        json.dumps(
            {
                "customizations": {
                    "vscode": {
                        "extensions": [extension_id],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    bundle_path = tmp_path / "bundle-local"
    extension_manager_module.export_offline_bundle(
        config_name=str(config_path),
        bundle_path=str(bundle_path),
        target_path=str(_artifact_cache_dir(tmp_path, "downloads")),
        code_path=code_binary,
    )

    sandbox_vscode_root = tmp_path / "vscode-server-local"
    sandbox_extensions = sandbox_vscode_root / "extensions"
    sandbox_extensions.mkdir(parents=True, exist_ok=True)
    (sandbox_extensions / "extensions.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(extension_manager_module, "vscode_root", sandbox_vscode_root)

    isolated_code_binary = _isolated_code_binary(
        code_binary, tmp_path / "isolated-code"
    )

    with _misconfigured_proxy(monkeypatch):
        extension_manager_module.import_offline_bundle(
            bundle_path=str(bundle_path),
            target_path=str(tmp_path / "import-cache-local"),
            code_path=isolated_code_binary,
            strict_offline=True,
            verify_manifest_signature=False,
        )

    installed = _installed_extensions(isolated_code_binary)
    assert extension_id in installed

    _uninstall_extension(isolated_code_binary, extension_id)
    installed_after_cleanup = _installed_extensions(isolated_code_binary)
    assert extension_id not in installed_after_cleanup
