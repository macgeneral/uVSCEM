#! /bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

# for parsing devcontainer.json (if it includes comments etc.)
import json5
import requests
from dependency_algorithm import Dependencies

from uvscem.api_client import CodeAPIManager
from uvscem.bundle_io import _as_string_list
from uvscem.code_manager import CodeManager
from uvscem.exceptions import (
    InstallationWorkflowError,
    UvscemError,
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
    _uvscem_version,
)
from uvscem.models import ExtensionSpec, ResolvedExtensionRequest
from uvscem.vsce_sign_bootstrap import (
    DEFAULT_VSCE_SIGN_VERSION,
    provision_vsce_sign_binary_for_run,
)
from uvscem.vscode_paths import resolve_vscode_root

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"


# Exposed as a module attribute so tests can read the configured limit.
max_retries: int = MAX_INSTALL_RETRIES
# VSCode extension installation directory (monkeypatched in tests via setattr)
vscode_root: Path = resolve_vscode_root()
logger: logging.Logger = logging.getLogger(__name__)

TRUSTED_DOWNLOAD_HOSTS: tuple[str, ...] = (
    "marketplace.visualstudio.com",
    "gallery.vsassets.io",
    "gallerycdn.vsassets.io",
    "visualstudio.com",
    "vsassets.io",
)


class InstalledIdentifier(TypedDict, total=False):
    id: str


class InstalledEntry(TypedDict, total=False):
    identifier: InstalledIdentifier
    version: str


MetadataEntry = dict[str, object]


def _as_installed_entry(value: object) -> InstalledEntry:
    if not isinstance(value, dict):
        return InstalledEntry()
    return cast(InstalledEntry, value)


class _SupportsNetworkOptions(Protocol):
    allow_unsigned: bool
    allow_untrusted_urls: bool
    allow_http: bool
    disable_ssl_verification: bool
    ca_bundle: str


def _workflow_error_message(operation: str, exc: Exception) -> str:
    return f"{operation} failed: {exc}"


def _sanitize_log_str(value: str) -> str:
    """Strip newline characters that could enable log-injection attacks."""
    return value.replace("\n", "\\n").replace("\r", "\\r")


def _is_trusted_download_host(host: str) -> bool:
    normalized_host = host.lower()
    for trusted in TRUSTED_DOWNLOAD_HOSTS:
        if normalized_host == trusted or normalized_host.endswith(f".{trusted}"):
            return True
    return False


def _validate_download_url(
    url: str,
    *,
    purpose: str,
    allow_untrusted_urls: bool = False,
    allow_http: bool = False,
) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme != "https":
        if not (allow_http and scheme == "http"):
            raise ValueError(f"{purpose} URL must use https: {url}")
    host = parsed.hostname or ""
    if not allow_untrusted_urls and not _is_trusted_download_host(host):
        raise ValueError(f"{purpose} URL host is not trusted: {host or url}")
    return url


def _configure_http_session(
    session: requests.Session | SimpleNamespace,
    *,
    disable_ssl_verification: bool,
    ca_bundle: str,
) -> None:
    if disable_ssl_verification:
        # Suppress per-request InsecureRequestWarning scoped to this session only.
        # We deliberately do NOT call urllib3.disable_warnings() to avoid silencing
        # warnings from other sessions in the same process.
        session.verify = False
        return
    if ca_bundle:
        ca_path = Path(ca_bundle)
        if not ca_path.is_file():
            raise ValueError(f"CA bundle path is not a readable file: {ca_bundle}")
        session.verify = ca_bundle


def _apply_network_options(
    manager: _SupportsNetworkOptions,
    *,
    allow_unsigned: bool,
    allow_untrusted_urls: bool,
    allow_http: bool,
    disable_ssl_verification: bool,
    ca_bundle: str,
) -> None:
    manager.allow_unsigned = allow_unsigned
    manager.allow_untrusted_urls = allow_untrusted_urls
    manager.allow_http = allow_http
    manager.disable_ssl_verification = disable_ssl_verification
    manager.ca_bundle = ca_bundle

    api_manager = getattr(manager, "api_manager", None)
    session = getattr(api_manager, "session", None)
    if session is not None:
        _configure_http_session(
            session,
            disable_ssl_verification=disable_ssl_verification,
            ca_bundle=ca_bundle,
        )


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract each member only after confirming its resolved path stays inside destination."""
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        member_path = destination.joinpath(member.filename)
        try:
            member_resolved = member_path.resolve()
            member_resolved.relative_to(destination_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Unsafe archive member path outside extraction root: {member.filename}"
            ) from exc
        archive.extract(member, destination)


def _extract_vsix_to_extension_dir(extension_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vscode-extension-extract.") as tmp_dir:
        with zipfile.ZipFile(extension_path, "r") as zip_obj:
            _safe_extract_zip(zip_obj, Path(tmp_dir))
            shutil.move(Path(tmp_dir, "extension/"), destination)


# Paths the resolved CLI extensions directory must not be rooted under.
# Writing extension files into system directories could corrupt the OS.
_UNTRUSTED_CLI_EXTENSION_DIR_PREFIXES: tuple[Path, ...] = (
    Path("/bin"),
    Path("/sbin"),
    Path("/etc"),
    Path("/usr/bin"),
    Path("/usr/sbin"),
    Path("/boot"),
    Path("/sys"),
    Path("/proc"),
    Path("/dev"),
)


def _resolve_cli_extensions_dir(code_binary: str) -> Path | None:
    code_path = Path(code_binary)
    if not code_path.is_file():
        return None

    try:
        script = code_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    match = re.search(
        r"--extensions-dir\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
        script,
    )
    if match is None:
        return None

    extensions_dir = next(group for group in match.groups() if group is not None)
    result = Path(extensions_dir).expanduser().resolve()
    for denied in _UNTRUSTED_CLI_EXTENSION_DIR_PREFIXES:
        if result == denied or str(result).startswith(str(denied) + os.sep):
            logger.warning(
                "CLI extensions dir resolved to a system path, ignoring: %s",
                result,
            )
            return None
    return result


def _write_json_atomic(path: Path, payload: object) -> None:
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
        try:
            json.dump(payload, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)


class CodeExtensionManager:
    """Proxy aware helper script to install VSCode extensions in a DevContainer."""

    api_manager: CodeAPIManager
    socket_manager: CodeManager
    code_binary: str
    dev_container_config_path: Path
    extensions: list[str]
    installed: list[InstalledEntry]
    extension_dependencies: dict[str, set[str]]
    extension_metadata: dict[str, MetadataEntry]
    target_path: Path
    vsce_sign_binary: Path | None
    allow_unsigned: bool
    allow_untrusted_urls: bool
    allow_http: bool
    disable_ssl_verification: bool
    ca_bundle: str

    def __init__(
        self,
        config_name: str = "devcontainer.json",
        code_path: str = "code",
        target_directory: str = "",
        allow_unsigned: bool = False,
        allow_untrusted_urls: bool = False,
        allow_http: bool = False,
        disable_ssl_verification: bool = False,
        ca_bundle: str = "",
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
            Path(os.path.expandvars(target_directory)).expanduser().absolute()
            if target_directory
            else Path.home().joinpath("cache/.vscode/extensions").absolute()
        )
        self.vsce_sign_binary = None
        self._vscode_root: Path = vscode_root
        _apply_network_options(
            self,
            allow_unsigned=allow_unsigned,
            allow_untrusted_urls=allow_untrusted_urls,
            allow_http=allow_http,
            disable_ssl_verification=disable_ssl_verification,
            ca_bundle=ca_bundle,
        )
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
        """Append each dependency that is not already present in extensions (mutates in place)."""
        for dependency in dependencies:
            if dependency not in extensions:
                extensions.append(dependency)

    async def _load_extension_specs(self) -> list[ExtensionSpec]:
        parsed_config: dict[str, object] = await asyncio.to_thread(
            lambda: json5.loads(self.dev_container_config_path.read_text())
        )
        customizations = parsed_config.get("customizations", {})
        if not isinstance(customizations, dict):
            customizations = {}
        vscode = customizations.get("vscode", {})
        if not isinstance(vscode, dict):
            vscode = {}
        extensions_value = vscode.get("extensions", [])
        raw_extensions: list[str] = (
            [item for item in extensions_value if isinstance(item, str)]
            if isinstance(extensions_value, list)
            else []
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
    ) -> MetadataEntry:
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

            dependencies = self.sanitize_dependencies(
                _as_string_list(metadata.get("dependencies", []))
            )
            self.extension_dependencies[extension_id].update(dependencies)
            self.add_missing_dependency(extensions, dependencies)

            # dont treat extension packs as dependencies
            extension_pack = self.sanitize_dependencies(
                _as_string_list(metadata.get("extension_pack", []))
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
        metadata = self.extension_metadata.get(extension_id, {})
        url = _validate_download_url(
            str(metadata.get("url", "")),
            purpose=f"Extension download for {extension_id}",
            allow_untrusted_urls=self.allow_untrusted_urls,
            allow_http=self.allow_http,
        )
        if not url:
            raise ValueError(f"Missing download URL for extension: {extension_id}")
        headers: dict = {
            "User-Agent": DEFAULT_USER_AGENT,
        }

        def _download_sync() -> Path:
            target_path = self.target_path.joinpath(self.get_filename(extension_id))
            if target_path.is_file() and target_path.stat().st_size > 0:
                if zipfile.is_zipfile(target_path):
                    logger.info(
                        "Reusing cached VSIX for %s: %s",
                        _sanitize_log_str(extension_id),
                        target_path,
                    )
                    return target_path
                logger.warning(
                    "Cached VSIX for %s is not a valid zip, re-downloading: %s",
                    _sanitize_log_str(extension_id),
                    target_path,
                )
            logger.info("Installing %s from %s", _sanitize_log_str(extension_id), url)
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
        metadata = self.extension_metadata.get(extension_id, {})
        url = _validate_download_url(
            str(metadata.get("signature", "")),
            purpose=f"Signature download for {extension_id}",
            allow_untrusted_urls=self.allow_untrusted_urls,
            allow_http=self.allow_http,
        )
        if not url:
            raise ValueError(f"Missing signature URL for extension: {extension_id}")
        headers: dict = {
            "User-Agent": DEFAULT_USER_AGENT,
        }

        def _download_sync() -> Path:
            target_path = self.target_path.joinpath(
                self.get_signature_filename(extension_id)
            )
            if target_path.is_file() and target_path.stat().st_size > 0:
                if zipfile.is_zipfile(target_path):
                    logger.info(
                        "Reusing cached signature archive for %s: %s",
                        _sanitize_log_str(extension_id),
                        target_path,
                    )
                    return target_path
                logger.warning(
                    "Cached signature archive for %s is not a valid zip, re-downloading: %s",
                    _sanitize_log_str(extension_id),
                    target_path,
                )
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
        self, extension_id: str, extension_path: Path, update_json: bool = True
    ) -> None:
        # steps:
        # [x] parse extensions.json for installed extensions -> self.installed
        # [x] unpack everything to ~/.vscode-server/extensions/extension.bundleid-version
        # [x] add installation_metadata to self.installed and export it to ~/.vscode-server/extensions/extensions.json
        # [x] repeat the steps for all extensions listed in extension_pack (check their dependencies!)
        target_path = self._vscode_root.joinpath(
            f"extensions/{self.get_dirname(extension_id)}"
        )

        def _install_manual_sync() -> None:
            _extract_vsix_to_extension_dir(extension_path, target_path)

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

        def _metadata_extension_id(metadata: InstalledEntry) -> str:
            return str(metadata.get("identifier", {}).get("id", "")).lower()

        for eid in extension_ids:
            installation_metadata = self.extension_metadata.get(eid, {}).get(
                "installation_metadata"
            )
            if installation_metadata:
                incoming = _as_installed_entry(installation_metadata)
                incoming_id = _metadata_extension_id(incoming)
                if not incoming_id:
                    self.installed.append(incoming)
                    continue

                replaced = False
                deduplicated: list[InstalledEntry] = []
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
            if not self.allow_unsigned:
                raise ValueError(
                    f"Missing signature metadata for extension {extension_id}; "
                    f"use --allow-unsigned to install anyway (INSECURE)"
                )
            msg = (
                f"SECURITY WARNING: Missing signature metadata for extension {extension_id}; "
                f"installing anyway because --allow-unsigned is set (INSECURE)"
            )
            logger.warning(msg)
            print(msg, file=sys.stderr)

        # installing extension packs doesn't work because the code install routine tries fetching other packages itself
        extension_pack_items = _as_string_list(metadata.get("extension_pack", []))
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
            attempt = retries
            while True:
                try:
                    await asyncio.to_thread(
                        run_code_cli_install,
                        code_binary=self.code_binary,
                        extension_path=file_path,
                        run_command=subprocess.run,
                    )
                    break
                except subprocess.CalledProcessError as exc:
                    logger.error(f"Something went wrong: {exc}")
                    await self.socket_manager.find_socket(update_environment=True)
                    if attempt >= max_retries:
                        logger.warning(
                            f"Code CLI installation failed for {extension_id}; falling back to manual VSIX extraction"
                        )
                        try:
                            await self.install_extension_manually(
                                extension_id,
                                file_path,
                                update_json=True,
                            )

                            cli_extensions_dir = _resolve_cli_extensions_dir(
                                self.code_binary
                            )
                            if cli_extensions_dir is not None:
                                cli_target = cli_extensions_dir.joinpath(
                                    self.get_dirname(extension_id)
                                )
                                await asyncio.to_thread(
                                    _extract_vsix_to_extension_dir,
                                    file_path,
                                    cli_target,
                                )
                            return
                        except (OSError, ValueError, zipfile.BadZipFile):
                            raise exc from exc
                    attempt += 1

            expected_extension_dir = self._vscode_root.joinpath(
                f"extensions/{self.get_dirname(extension_id)}"
            )
            if not expected_extension_dir.is_dir():
                if zipfile.is_zipfile(file_path):
                    await self.install_extension_manually(
                        extension_id,
                        file_path,
                        update_json=True,
                    )
                else:
                    logger.warning(
                        f"Installed VSIX for {extension_id} is not a zip archive, skipping managed extraction"
                    )

    async def find_installed(self) -> list[InstalledEntry]:
        """Return a list of already installed extensions."""

        def _find_sync() -> list[InstalledEntry]:
            extensions_path: Path = self.extensions_json_path()
            if not extensions_path.is_file():
                return []
            with open(extensions_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, list):
                    return []
                return [
                    cast(InstalledEntry, item)
                    for item in loaded
                    if isinstance(item, dict)
                ]

        return await asyncio.to_thread(_find_sync)

    def extensions_json_path(self) -> Path:
        return self._vscode_root.joinpath("extensions/extensions.json")

    async def exclude_installed(self) -> None:
        """Remove all already installed extensions from the extensions list."""
        installed_pairs: set[tuple[str, str]] = set()
        for installed_extension in self.installed:
            installed_name = str(
                installed_extension.get("identifier", {}).get("id", "")
            ).lower()
            installed_version = str(installed_extension.get("version", ""))
            if installed_name and installed_version:
                installed_pairs.add((installed_name, installed_version))

        filtered_extensions: list[str] = []
        for extension in self.extensions:
            extension_name = extension.lower()
            extension_version = str(
                self.extension_metadata.get(extension, {}).get("version", "")
            )
            if (extension_name, extension_version) in installed_pairs:
                logger.info(
                    f"Skipping {extension} ({extension_version}), already installed."
                )
                continue
            filtered_extensions.append(extension)

        self.extensions = filtered_extensions

    async def install_async(self) -> None:
        """Asynchronously install all configured extensions."""
        await self.exclude_installed()

        provisioner = provision_vsce_sign_binary_for_run(
            install_dir=self.target_path,
            session=self.api_manager.session,
        )
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
    target_path: str = "",
    log_level: str = "info",
    allow_unsigned: bool = False,
    allow_untrusted_urls: bool = False,
    allow_http: bool = False,
    disable_ssl_verification: bool = False,
    ca_bundle: str = "",
) -> None:
    """Install all extensions listed in devcontainer.json."""
    logging.basicConfig(
        level=(getattr(logging, log_level.upper())),
        format="%(relativeCreated)d [%(levelname)s] %(message)s",
    )
    logger.info("Attempting to install all necessary DevContainer extensions.")

    config_path = Path(config_name)
    resolved_config_path = (
        config_path
        if config_path.is_absolute()
        else Path.cwd().joinpath(config_path).absolute()
    )
    if not resolved_config_path.is_file():
        raise InstallationWorkflowError(
            f"DevContainer configuration not found: {resolved_config_path}"
        )

    async def _run() -> None:
        manager = CodeExtensionManager(
            config_name=config_name,
            code_path=code_path,
            target_directory=target_path,
            allow_unsigned=allow_unsigned,
            allow_untrusted_urls=allow_untrusted_urls,
            allow_http=allow_http,
            disable_ssl_verification=disable_ssl_verification,
            ca_bundle=ca_bundle,
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


# Re-export bundle workflow functions so existing call sites (tests, CLI) keep working.
from uvscem.bundle_workflow import (
    export_offline_bundle,
    import_offline_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser while keeping existing install option names."""
    parser = argparse.ArgumentParser(prog="uvscem")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="install",
        choices=["install", "export", "import"],
    )
    parser.add_argument("--config-name", default="devcontainer.json")
    parser.add_argument("--code-path", default="code")
    parser.add_argument("--target-path", default="")
    parser.add_argument("--bundle-path", default="./uvscem-offline-bundle")
    parser.add_argument("--vsce-sign-version", default=DEFAULT_VSCE_SIGN_VERSION)
    parser.add_argument("--vsce-sign-targets", default="current")
    parser.add_argument("--manifest-signing-key", default="")
    parser.add_argument("--strict-offline", action="store_true")
    parser.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="Allow installation when signature metadata is missing (INSECURE).",
    )
    parser.add_argument(
        "--allow-untrusted-urls",
        action="store_true",
        help="Allow download hosts outside the trusted marketplace allowlist (INSECURE).",
    )
    parser.add_argument(
        "--allow-http",
        action="store_true",
        help="Allow http:// URLs in addition to https:// (INSECURE). Useful for SSL-breaking proxies.",
    )
    parser.add_argument(
        "--disable-ssl-verification",
        action="store_true",
        help="Disable TLS certificate verification for HTTP requests (INSECURE).",
    )
    parser.add_argument(
        "--ca-bundle",
        default="",
        help="Path to custom CA bundle for TLS verification (overrides default trust store).",
    )
    parser.add_argument("--skip-manifest-signature-verification", action="store_true")
    parser.add_argument("--manifest-verification-keyring", default="")
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "version", False):
        print(f"uvscem {_uvscem_version}")
        print(f"User-Agent: {DEFAULT_USER_AGENT}")
        return

    command = getattr(args, "command", "install")

    if command == "install":
        try:
            install(
                config_name=args.config_name,
                code_path=args.code_path,
                target_path=args.target_path,
                log_level=args.log_level,
                allow_unsigned=getattr(args, "allow_unsigned", False),
                allow_untrusted_urls=getattr(args, "allow_untrusted_urls", False),
                allow_http=getattr(args, "allow_http", False),
                disable_ssl_verification=getattr(
                    args, "disable_ssl_verification", False
                ),
                ca_bundle=getattr(args, "ca_bundle", ""),
            )
        except InstallationWorkflowError as exc:
            logger.error(exc)
            sys.exit(1)
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            logger.debug("Traceback:", exc_info=True)
            sys.exit(1)
        return

    if command == "export":
        try:
            export_offline_bundle(
                config_name=args.config_name,
                bundle_path=args.bundle_path,
                target_path=args.target_path,
                code_path=args.code_path,
                log_level=args.log_level,
                vsce_sign_version=args.vsce_sign_version,
                vsce_sign_targets=getattr(args, "vsce_sign_targets", "current"),
                manifest_signing_key=getattr(args, "manifest_signing_key", ""),
                allow_untrusted_urls=getattr(args, "allow_untrusted_urls", False),
                allow_http=getattr(args, "allow_http", False),
                disable_ssl_verification=getattr(
                    args, "disable_ssl_verification", False
                ),
                ca_bundle=getattr(args, "ca_bundle", ""),
            )
        except UvscemError as exc:
            logger.error(exc)
            sys.exit(1)
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            logger.debug("Traceback:", exc_info=True)
            sys.exit(1)
        return

    try:
        verify_manifest_signature = not getattr(
            args, "skip_manifest_signature_verification", False
        )

        import_offline_bundle(
            bundle_path=args.bundle_path,
            code_path=args.code_path,
            target_path=args.target_path,
            log_level=args.log_level,
            strict_offline=getattr(args, "strict_offline", False),
            verify_manifest_signature=verify_manifest_signature,
            manifest_verification_keyring=getattr(
                args, "manifest_verification_keyring", ""
            ),
        )
    except UvscemError as exc:
        logger.error(exc)
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
