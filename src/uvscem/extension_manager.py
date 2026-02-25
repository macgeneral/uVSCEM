#! /bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# for parsing devcontainer.json (if it includes comments etc.)
import json5
import requests
from dependency_algorithm import Dependencies

from uvscem.api_client import CodeAPIManager
from uvscem.bundle_io import (
    build_bundle_manifest,
    build_extension_manifest_entry,
    resolve_bundled_vsce_sign_binary,
    resolve_vsce_sign_targets,
    sha256_file,
)
from uvscem.code_manager import CodeManager
from uvscem.exceptions import (
    InstallationWorkflowError,
    OfflineBundleExportError,
    OfflineBundleImportExecutionError,
    OfflineBundleImportMissingFileError,
    OfflineBundleImportValidationError,
    OfflineModeError,
)
from uvscem.install_engine import (
    run_code_cli_install,
    run_vsce_sign_verify,
    stream_download_to_target,
)
from uvscem.internal_config import (
    DEFAULT_USER_AGENT,
    HTTP_STREAM_CONNECT_TIMEOUT_SECONDS,
    HTTP_STREAM_READ_TIMEOUT_SECONDS,
    MAX_INSTALL_RETRIES,
)
from uvscem.models import ExtensionSpec, ResolvedExtensionRequest
from uvscem.vsce_sign_bootstrap import (
    DEFAULT_VSCE_SIGN_VERSION,
    SUPPORTED_VSCE_SIGN_TARGETS,
    get_vsce_sign_package_name,
    get_vsce_sign_target,
    install_vsce_sign_binary_for_target,
    provision_vsce_sign_binary_for_run,
)
from uvscem.vscode_paths import resolve_vscode_root

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"


# attempt to install an extension a maximum of three times
max_retries = MAX_INSTALL_RETRIES
user_agent: str = DEFAULT_USER_AGENT
# VSCode extension installation directory
vscode_root: Path = resolve_vscode_root()
logger: logging.Logger = logging.getLogger(__name__)


def _workflow_error_message(operation: str, exc: Exception) -> str:
    return f"{operation} failed: {exc}"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(payload, tmp_file)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    tmp_path.replace(path)


class CodeExtensionManager(object):
    """Proxy aware helper script to install VSCode extensions in a DevContainer."""

    api_manager: CodeAPIManager
    socket_manager: CodeManager
    code_binary: str
    dev_container_config_path: Path
    extensions: list[str]
    installed: list[dict[str, Any]]
    extension_dependencies: dict[str, set[str]]
    extension_metadata: dict[str, dict[str, Any]]
    target_path: Path
    vsce_sign_binary: Path | None

    def __init__(
        self,
        config_name: str = "devcontainer.json",
        code_path: str = "code",
        target_directory: str = "",
    ) -> None:
        """Set up the CodeExtensionManager."""
        self.extension_dependencies = defaultdict(set)
        self.extension_metadata = {}
        self.api_manager = CodeAPIManager()
        self.socket_manager = CodeManager()
        self.code_binary = code_path
        config_path = Path(config_name)
        self.dev_container_config_path = (
            config_path
            if config_path.is_absolute()
            else Path.cwd()
            .joinpath(
                config_path,
            )
            .absolute()
        )
        self.extensions = []
        self.installed = []
        self.target_path = (
            Path(target_directory).absolute()
            if target_directory
            else Path.home().joinpath("cache/.vscode/extensions").absolute()
        )
        self.vsce_sign_binary = None
        # create target directory if necessary
        self.target_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Asynchronously initialize runtime state before installation."""
        await self.socket_manager.initialize()
        self.extensions = await self.parse_all_extensions()
        self.installed = await self.find_installed()

    def sanitize_dependencies(self, extensions: list[str]) -> list[str]:
        """
        Remove VSCode internal extensions which are not available
        on the Marketplace, but still are listed as dependency.
        """
        sanitized = []

        for e in extensions:
            if not e.startswith("vscode."):
                sanitized.append(e)

        return sanitized

    def add_missing_dependency(
        self, extensions: list[str], dependencies: list[str]
    ) -> None:
        for dependency in dependencies:
            if dependency not in extensions:
                extensions.append(dependency)

    async def _load_extension_specs(self) -> list[ExtensionSpec]:
        parsed_config: dict[str, Any] = await asyncio.to_thread(
            lambda: json5.loads(self.dev_container_config_path.read_text())
        )
        raw_extensions: list[str] = list(
            parsed_config.get("customizations", {})
            .get("vscode", {})
            .get("extensions", [])
        )
        return [
            ExtensionSpec(*self.parse_extension_spec(extension_spec))
            for extension_spec in raw_extensions
        ]

    async def _resolve_extension_requests(
        self, specs: list[ExtensionSpec]
    ) -> tuple[list[str], dict[str, str]]:
        extensions: list[str] = []
        extension_pins: dict[str, str] = {}
        for spec in specs:
            if not spec.extension_id:
                continue
            extensions.append(spec.extension_id)
            if spec.pinned_version:
                extension_pins[spec.extension_id] = spec.pinned_version
        return extensions, extension_pins

    async def _fetch_metadata_for_request(
        self,
        request: ResolvedExtensionRequest,
    ) -> dict[str, Any]:
        metadata = await self.api_manager.get_extension_metadata(
            request.extension_id,
            include_latest_stable_version_only=not bool(request.pinned_version),
            requested_version=request.pinned_version,
        )
        versions = metadata.get(request.extension_id, [])
        if not versions:
            if request.pinned_version:
                raise ValueError(
                    f"Pinned extension version not found: {request.extension_id}@{request.pinned_version}"
                )
            return {}
        return dict(versions[0])

    async def parse_all_extensions(self) -> list[str]:
        specs = await self._load_extension_specs()
        extensions, extension_pins = await self._resolve_extension_requests(specs)

        for extension_id in extensions:
            request = ResolvedExtensionRequest(
                extension_id=extension_id,
                pinned_version=extension_pins.get(extension_id, ""),
            )
            metadata = await self._fetch_metadata_for_request(request)
            if not metadata:
                continue
            self.extension_metadata[extension_id] = metadata

            dependencies = self.sanitize_dependencies(metadata.get("dependencies", []))
            self.extension_dependencies[extension_id].update(dependencies)
            self.add_missing_dependency(extensions, dependencies)

            # dont treat extension packs as dependencies
            extension_pack = self.sanitize_dependencies(
                metadata.get("extension_pack", [])
            )
            self.add_missing_dependency(extensions, extension_pack)

        # get installation order
        dependencies = Dependencies(self.extension_dependencies)
        return dependencies.resolve_dependencies()

    def parse_extension_spec(self, extension_spec: str) -> tuple[str, str]:
        spec = extension_spec.strip()
        if not spec:
            return "", ""
        if "@" not in spec:
            return spec, ""
        extension_id, _, requested_version = spec.rpartition("@")
        extension_id = extension_id.strip()
        requested_version = requested_version.strip()
        if not extension_id or not requested_version:
            return spec, ""
        return extension_id, requested_version

    def get_dirname(self, extension_id: str) -> str:
        version: str = str(
            self.extension_metadata.get(extension_id, {}).get("version", "")
        )
        return f"{extension_id}-{version}"

    def get_filename(self, extension_id: str) -> str:
        return f"{self.get_dirname(extension_id)}.vsix"

    def get_signature_filename(self, extension_id: str) -> str:
        return f"{self.get_dirname(extension_id)}.sigzip"

    async def download_extension(self, extension_id: str) -> Path:
        """Download an extension and return the downloaded file's path."""
        metadata: dict = self.extension_metadata.get(extension_id, {})
        url: str = str(metadata.get("url", ""))
        if not url:
            raise ValueError(f"Missing download URL for extension: {extension_id}")
        headers: dict = {
            "User-Agent": user_agent,
        }

        def _download_sync() -> Path:
            target_path = self.target_path.joinpath(self.get_filename(extension_id))
            if target_path.is_file() and target_path.stat().st_size > 0:
                logger.info(f"Reusing cached VSIX for {extension_id}: {target_path}")
                return target_path
            logger.info(f"Installing {extension_id} from {url}")
            return stream_download_to_target(
                session=self.api_manager.session,
                url=url,
                target_path=target_path,
                headers=headers,
                temp_prefix="vscode-extension.",
                timeout=(
                    HTTP_STREAM_CONNECT_TIMEOUT_SECONDS,
                    HTTP_STREAM_READ_TIMEOUT_SECONDS,
                ),
            )

        return await asyncio.to_thread(_download_sync)

    async def download_signature_archive(self, extension_id: str) -> Path:
        """Download an extension signature archive and return the downloaded file path."""
        metadata: dict = self.extension_metadata.get(extension_id, {})
        url: str = str(metadata.get("signature", ""))
        if not url:
            raise ValueError(f"Missing signature URL for extension: {extension_id}")
        headers: dict = {
            "User-Agent": user_agent,
        }

        def _download_sync() -> Path:
            target_path = self.target_path.joinpath(
                self.get_signature_filename(extension_id)
            )
            if target_path.is_file() and target_path.stat().st_size > 0:
                logger.info(
                    f"Reusing cached signature archive for {extension_id}: {target_path}"
                )
                return target_path
            return stream_download_to_target(
                session=self.api_manager.session,
                url=url,
                target_path=target_path,
                headers=headers,
                temp_prefix="vscode-extension-signature.",
                timeout=(
                    HTTP_STREAM_CONNECT_TIMEOUT_SECONDS,
                    HTTP_STREAM_READ_TIMEOUT_SECONDS,
                ),
            )

        return await asyncio.to_thread(_download_sync)

    async def verify_extension_signature(
        self, extension_path: Path, signature_archive_path: Path
    ) -> None:
        """Verify one VSIX against its signature archive using vsce-sign."""
        if self.vsce_sign_binary is None:
            raise RuntimeError("vsce-sign binary is not initialized for this run")

        await asyncio.to_thread(
            run_vsce_sign_verify,
            vsce_sign_binary=self.vsce_sign_binary,
            extension_path=extension_path,
            signature_archive_path=signature_archive_path,
            run_command=subprocess.run,
        )

    async def install_extension_manually(
        self, extension_id: str, extension_path: Path, update_json=True
    ) -> None:
        # steps:
        # [x] parse extensions.json for installed extensions -> self.installed
        # [x] unpack everything to ~/.vscode-server/extensions/extension.bundleid-version
        # [x] add installation_metadata to self.installed and export it to ~/.vscode-server/extensions/extensions.json
        # [x] repeat the steps for all extensions listed in extension_pack (check their dependencies!)
        target_path = vscode_root.joinpath(
            f"extensions/{self.get_dirname(extension_id)}"
        )

        def _install_manual_sync() -> None:
            if target_path.exists():
                shutil.rmtree(target_path)

            with tempfile.TemporaryDirectory(
                prefix="vscode-extension-extract."
            ) as tmp_dir:
                with zipfile.ZipFile(extension_path, "r") as zip_obj:
                    zip_obj.extractall(tmp_dir)
                    shutil.move(Path(tmp_dir, "extension/"), target_path)

        await asyncio.to_thread(_install_manual_sync)

        if update_json:
            await self.update_extensions_json(extension_id=extension_id)

    async def update_extensions_json(
        self, extension_id: str = "", extension_ids: list[str] | None = None
    ):
        self.installed = await self.find_installed()
        extension_ids = list(extension_ids or [])

        if extension_id:
            extension_ids.append(extension_id)

        def _metadata_extension_id(metadata: dict[str, Any]) -> str:
            return str(metadata.get("identifier", {}).get("id", "")).lower()

        for eid in extension_ids:
            installation_metadata = self.extension_metadata.get(eid, {}).get(
                "installation_metadata"
            )
            if installation_metadata:
                incoming = dict(installation_metadata)
                incoming_id = _metadata_extension_id(incoming)
                if not incoming_id:
                    self.installed.append(incoming)
                    continue

                replaced = False
                deduplicated: list[dict[str, Any]] = []
                for item in self.installed:
                    if _metadata_extension_id(item) == incoming_id:
                        if not replaced:
                            deduplicated.append(incoming)
                            replaced = True
                        continue
                    deduplicated.append(item)

                if not replaced:
                    deduplicated.append(incoming)

                self.installed = deduplicated

        def _update_sync() -> None:
            json_path: Path = self.extensions_json_path()
            _write_json_atomic(json_path, self.installed)

        await asyncio.to_thread(_update_sync)

    async def install_extension(
        self, extension_id: str, extension_pack: bool = False, retries: int = 0
    ) -> None:
        """Install a single extension."""
        file_name = self.get_filename(extension_id)
        file_path = self.target_path.joinpath(file_name)
        metadata = self.extension_metadata.get(extension_id, {})

        if file_path.is_file():
            logger.info(f"File {file_name} exists - skipping download...")
        else:
            file_path = await self.download_extension(extension_id)

        signature_url: str = str(metadata.get("signature", ""))
        if signature_url:
            signature_file = self.target_path.joinpath(
                self.get_signature_filename(extension_id)
            )
            if not signature_file.is_file():
                signature_file = await self.download_signature_archive(extension_id)
            await self.verify_extension_signature(file_path, signature_file)
        else:
            logger.warning(
                f"Missing signature metadata for extension {extension_id}, skipping verification"
            )

        # installing extension packs doesn't work because the code install routine tries fetching other packages itself
        extension_pack_items = list(metadata.get("extension_pack", []))
        if extension_pack_items:
            manually_installed = []

            if not extension_pack:
                manually_installed.append(extension_id)
                extension_pack = True

            # also repeat this step for the packages listed in extension pack
            for ep in extension_pack_items:
                if ep in self.extensions:
                    await self.install_extension(
                        self.extensions.pop(self.extensions.index(ep)),
                        extension_pack=True,
                    )
                manually_installed.append(ep)

            await self.update_extensions_json(extension_ids=manually_installed)

        if extension_pack:
            await self.install_extension_manually(
                extension_id, file_path, update_json=False
            )
        else:
            try:
                await asyncio.to_thread(
                    run_code_cli_install,
                    code_binary=self.code_binary,
                    extension_path=file_path,
                    run_command=subprocess.run,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Something went wrong: {e}")
                await self.socket_manager.find_socket(update_environment=True)
                if retries < max_retries:
                    await self.install_extension(
                        extension_id,
                        extension_pack=extension_pack,
                        retries=retries + 1,
                    )

    async def find_installed(self) -> list[dict[str, Any]]:
        """Return a list of already installed extensions."""

        def _find_sync() -> list[dict[str, Any]]:
            extensions_path: Path = self.extensions_json_path()
            with open(extensions_path, "r") as f:
                return json.load(f)

        return await asyncio.to_thread(_find_sync)

    def extensions_json_path(self) -> Path:
        return vscode_root.joinpath("extensions/extensions.json")

    async def exclude_installed(self) -> None:
        """Remove all already installed extensions from the extensions list."""
        for installed_extension in self.installed:
            for extension in self.extensions:
                installed_version: str = str(installed_extension.get("version", ""))
                installed_name: str = str(
                    installed_extension.get("identifier", {}).get("id", "")
                ).lower()
                extension_name: str = extension.lower()
                extension_version: str = str(
                    self.extension_metadata.get(extension, {}).get("version", "")
                )
                if (
                    extension_name == installed_name
                    and installed_version == extension_version
                ):
                    self.extensions.remove(extension)
                    logger.info(
                        f"Skipping {extension} ({extension_version}), already installed."
                    )

    async def install_async(self) -> None:
        """Asynchronously install all configured extensions."""
        await self.exclude_installed()

        provisioner = provision_vsce_sign_binary_for_run(install_dir=self.target_path)
        with provisioner as vsce_sign_binary:
            self.vsce_sign_binary = vsce_sign_binary
            try:
                while self.extensions:
                    await self.install_extension(self.extensions.pop(0))
                    await asyncio.sleep(1)
            finally:
                self.vsce_sign_binary = None


def install(
    config_name: str = "devcontainer.json",
    code_path: str = "code",
    target_path: str = "$HOME/cache/.vscode/extensions",
    log_level: str = "info",
) -> None:
    """Install all extensions listed in devcontainer.json."""
    logging.basicConfig(
        level=(getattr(logging, log_level.upper())),
        format="%(relativeCreated)d [%(levelname)s] %(message)s",
    )
    logger.info("Attempting to install all necessary DevContainer extensions.")

    async def _run() -> None:
        manager = CodeExtensionManager(
            config_name=config_name,
            code_path=code_path,
            target_directory=target_path,
        )
        await manager.initialize()
        await manager.install_async()

    try:
        asyncio.run(_run())
    except (
        requests.RequestException,
        subprocess.CalledProcessError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise InstallationWorkflowError(
            _workflow_error_message("Extension installation", exc)
        ) from exc


def _run_command(cmd: list[str]) -> None:
    process = subprocess.run(
        cmd,
        capture_output=True,
        check=False,
        text=True,
    )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            process.stdout,
            process.stderr,
        )


def _sign_bundle_manifest(manifest_path: Path, signing_key: str) -> Path:
    signature_path = manifest_path.with_suffix(f"{manifest_path.suffix}.asc")
    _run_command(
        [
            "gpg",
            "--batch",
            "--yes",
            "--armor",
            "--local-user",
            signing_key,
            "--output",
            str(signature_path),
            "--detach-sign",
            str(manifest_path),
        ]
    )
    return signature_path


def _verify_bundle_manifest_signature(
    manifest_path: Path,
    signature_path: Path,
    verification_keyring: str = "",
) -> None:
    cmd = ["gpg", "--batch"]
    if verification_keyring:
        cmd.extend(
            [
                "--no-default-keyring",
                "--keyring",
                verification_keyring,
            ]
        )
    cmd.extend(["--verify", str(signature_path), str(manifest_path)])
    _run_command(cmd)


def export_offline_bundle(
    config_name: str = "devcontainer.json",
    bundle_path: str = "./uvscem-offline-bundle",
    target_path: str = "$HOME/cache/.vscode/extensions",
    code_path: str = "code",
    log_level: str = "info",
    vsce_sign_version: str = DEFAULT_VSCE_SIGN_VERSION,
    vsce_sign_targets: str = "current",
    manifest_signing_key: str = "",
) -> None:
    """Export extensions, signatures, and vsce-sign into an offline bundle."""
    logging.basicConfig(
        level=(getattr(logging, log_level.upper())),
        format="%(relativeCreated)d [%(levelname)s] %(message)s",
    )

    try:
        bundle_dir = Path(bundle_path).expanduser().resolve()
        artifacts_dir = bundle_dir.joinpath("artifacts")
        cache_dir = Path(os.path.expandvars(target_path)).expanduser().resolve()
        vsce_sign_dir = bundle_dir.joinpath("vsce-sign")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vsce_sign_dir.mkdir(parents=True, exist_ok=True)

        resolved_targets = resolve_vsce_sign_targets(
            value=vsce_sign_targets,
            current_target=get_vsce_sign_target(),
            supported_targets=SUPPORTED_VSCE_SIGN_TARGETS,
        )

        async def _run() -> tuple[
            list[str],
            dict[str, dict[str, Any]],
            dict[str, tuple[Path, Path]],
        ]:
            manager = CodeExtensionManager(
                config_name=config_name,
                code_path=code_path,
                target_directory=str(cache_dir),
            )
            extensions = await manager.parse_all_extensions()
            downloaded: dict[str, tuple[Path, Path]] = {}
            for extension_id in extensions:
                vsix_path = await manager.download_extension(extension_id)
                signature_path = await manager.download_signature_archive(extension_id)
                downloaded[extension_id] = (vsix_path, signature_path)
            return extensions, manager.extension_metadata, downloaded

        extensions, extension_metadata, downloaded_artifacts = asyncio.run(_run())

        for extension_id in extensions:
            vsix_path, signature_path = downloaded_artifacts[extension_id]
            shutil.copy2(vsix_path, artifacts_dir.joinpath(vsix_path.name))
            shutil.copy2(signature_path, artifacts_dir.joinpath(signature_path.name))

        vsce_sign_binaries: list[dict[str, str]] = []
        for target in resolved_targets:
            target_dir = vsce_sign_dir.joinpath(target)
            binary_path = install_vsce_sign_binary_for_target(
                target=target,
                install_dir=target_dir,
                version=vsce_sign_version,
                force=True,
            )
            vsce_sign_binaries.append(
                {
                    "target": target,
                    "package": get_vsce_sign_package_name(target),
                    "binary": str(binary_path.relative_to(bundle_dir)),
                    "sha256": sha256_file(binary_path),
                }
            )

        extension_entries = [
            build_extension_manifest_entry(
                extension_id=extension_id,
                metadata=extension_metadata.get(extension_id, {}),
                artifacts_dir=artifacts_dir,
            )
            for extension_id in extensions
        ]

        manifest = build_bundle_manifest(
            config_name=config_name,
            ordered_extensions=extensions,
            extension_entries=extension_entries,
            vsce_sign_version=vsce_sign_version,
            vsce_sign_binaries=vsce_sign_binaries,
        )
        manifest_path = bundle_dir.joinpath("manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if manifest_signing_key:
            _sign_bundle_manifest(manifest_path, manifest_signing_key)
    except (
        requests.RequestException,
        subprocess.CalledProcessError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise OfflineBundleExportError(
            _workflow_error_message("Offline bundle export", exc)
        ) from exc


def import_offline_bundle(
    bundle_path: str,
    code_path: str = "code",
    target_path: str = "$HOME/cache/.vscode/extensions",
    log_level: str = "info",
    strict_offline: bool = False,
    verify_manifest_signature: bool = False,
    manifest_verification_keyring: str = "",
) -> None:
    """Install extensions from a previously exported offline bundle."""
    logging.basicConfig(
        level=(getattr(logging, log_level.upper())),
        format="%(relativeCreated)d [%(levelname)s] %(message)s",
    )

    try:
        bundle_dir = Path(bundle_path).expanduser().resolve()
        manifest_path = bundle_dir.joinpath("manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Offline bundle manifest not found: {manifest_path}"
            )

        if verify_manifest_signature:
            signature_path = manifest_path.with_suffix(f"{manifest_path.suffix}.asc")
            if not signature_path.is_file():
                raise FileNotFoundError(
                    f"Offline bundle manifest signature not found: {signature_path}"
                )
            _verify_bundle_manifest_signature(
                manifest_path=manifest_path,
                signature_path=signature_path,
                verification_keyring=manifest_verification_keyring,
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported offline bundle schema version")

        manager = CodeExtensionManager(
            config_name=str(manifest_path),
            code_path=code_path,
            target_directory=target_path,
        )

        if strict_offline and hasattr(manager, "api_manager"):
            session = getattr(manager.api_manager, "session", None)
            if session is not None:

                def _offline_request(*_args, **_kwargs):
                    raise RuntimeError(
                        "Network access is disabled in strict offline mode"
                    )

                setattr(session, "request", _offline_request)

        extensions_json = manager.extensions_json_path()
        extensions_json.parent.mkdir(parents=True, exist_ok=True)
        if not extensions_json.is_file():
            extensions_json.write_text("[]", encoding="utf-8")

        extension_entries = {
            str(entry.get("id", "")): entry for entry in manifest.get("extensions", [])
        }
        ordered_extensions: list[str] = [
            str(extension_id) for extension_id in manifest.get("ordered_extensions", [])
        ]

        artifacts_dir = bundle_dir.joinpath("artifacts")
        for extension_id in ordered_extensions:
            entry = extension_entries.get(extension_id)
            if not entry:
                raise ValueError(
                    f"Missing extension manifest entry for extension: {extension_id}"
                )

            filename = str(entry.get("filename", ""))
            signature_filename = str(entry.get("signature_filename", ""))
            source_vsix = artifacts_dir.joinpath(filename)
            source_signature = artifacts_dir.joinpath(signature_filename)
            if not source_vsix.is_file() or not source_signature.is_file():
                raise FileNotFoundError(
                    f"Missing offline artifact(s) for {extension_id}: {filename}, {signature_filename}"
                )

            expected_vsix_sha256 = str(entry.get("vsix_sha256", ""))
            if (
                expected_vsix_sha256
                and sha256_file(source_vsix) != expected_vsix_sha256
            ):
                raise ValueError(f"VSIX checksum mismatch for {extension_id}")

            expected_signature_sha256 = str(entry.get("signature_sha256", ""))
            if (
                expected_signature_sha256
                and sha256_file(source_signature) != expected_signature_sha256
            ):
                raise ValueError(f"Signature checksum mismatch for {extension_id}")

            shutil.copy2(source_vsix, manager.target_path.joinpath(filename))
            shutil.copy2(
                source_signature,
                manager.target_path.joinpath(signature_filename),
            )

            manager.extension_metadata[extension_id] = {
                "version": str(entry.get("version", "")),
                "url": "offline",
                "signature": "offline",
                "installation_metadata": entry.get("installation_metadata", {}),
                "dependencies": list(entry.get("dependencies", [])),
                "extension_pack": list(entry.get("extension_pack", [])),
            }

        vsce_sign_info = manifest.get("vsce_sign", {})
        vsce_sign_binary, expected_vsce_sign_sha256 = resolve_bundled_vsce_sign_binary(
            bundle_dir=bundle_dir,
            vsce_sign_info=vsce_sign_info,
            current_target=get_vsce_sign_target(),
        )

        if not vsce_sign_binary.is_file():
            raise FileNotFoundError(
                f"Bundled vsce-sign binary not found: {vsce_sign_binary}"
            )
        if (
            expected_vsce_sign_sha256
            and sha256_file(vsce_sign_binary) != expected_vsce_sign_sha256
        ):
            raise ValueError("Bundled vsce-sign checksum mismatch")

        async def _run() -> None:
            manager.installed = await manager.find_installed()
            manager.extensions = list(ordered_extensions)
            await manager.exclude_installed()

            manager.vsce_sign_binary = vsce_sign_binary
            try:
                while manager.extensions:
                    await manager.install_extension(manager.extensions.pop(0))
                    await asyncio.sleep(1)
            finally:
                manager.vsce_sign_binary = None

        asyncio.run(_run())
    except FileNotFoundError as exc:
        raise OfflineBundleImportMissingFileError(
            _workflow_error_message("Offline bundle import", exc)
        ) from exc
    except ValueError as exc:
        raise OfflineBundleImportValidationError(
            _workflow_error_message("Offline bundle import", exc)
        ) from exc
    except RuntimeError as exc:
        if "strict offline mode" in str(exc):
            raise OfflineModeError(
                _workflow_error_message("Offline bundle import", exc)
            ) from exc
        raise OfflineBundleImportExecutionError(
            _workflow_error_message("Offline bundle import", exc)
        ) from exc
    except (requests.RequestException, subprocess.CalledProcessError, OSError) as exc:
        raise OfflineBundleImportExecutionError(
            _workflow_error_message("Offline bundle import", exc)
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser while keeping existing install option names."""
    parser = argparse.ArgumentParser(prog="uvscem")
    parser.add_argument(
        "command",
        nargs="?",
        default="install",
        choices=["install", "export", "import"],
    )
    parser.add_argument("--config-name", default="devcontainer.json")
    parser.add_argument("--code-path", default="code")
    parser.add_argument("--target-path", default="$HOME/cache/.vscode/extensions")
    parser.add_argument("--bundle-path", default="./uvscem-offline-bundle")
    parser.add_argument("--vsce-sign-version", default=DEFAULT_VSCE_SIGN_VERSION)
    parser.add_argument("--vsce-sign-targets", default="current")
    parser.add_argument("--manifest-signing-key", default="")
    parser.add_argument("--strict-offline", action="store_true")
    parser.add_argument("--verify-manifest-signature", action="store_true")
    parser.add_argument("--manifest-verification-keyring", default="")
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = getattr(args, "command", "install")

    if command == "install":
        install(
            config_name=args.config_name,
            code_path=args.code_path,
            target_path=args.target_path,
            log_level=args.log_level,
        )
        return

    if command == "export":
        export_offline_bundle(
            config_name=args.config_name,
            bundle_path=args.bundle_path,
            target_path=args.target_path,
            code_path=args.code_path,
            log_level=args.log_level,
            vsce_sign_version=args.vsce_sign_version,
            vsce_sign_targets=getattr(args, "vsce_sign_targets", "current"),
            manifest_signing_key=getattr(args, "manifest_signing_key", ""),
        )
        return

    import_offline_bundle(
        bundle_path=args.bundle_path,
        code_path=args.code_path,
        target_path=args.target_path,
        log_level=args.log_level,
        strict_offline=getattr(args, "strict_offline", False),
        verify_manifest_signature=getattr(args, "verify_manifest_signature", False),
        manifest_verification_keyring=getattr(
            args, "manifest_verification_keyring", ""
        ),
    )


if __name__ == "__main__":
    main()
