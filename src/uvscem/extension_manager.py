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
from uvscem.code_manager import CodeManager
from uvscem.vsce_sign_bootstrap import provision_vsce_sign_binary_for_run

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"


# attempt to install an extension a maximum of three times
max_retries = 3
user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15"
# VSCode extension installation directory
vscode_root: Path = Path.home().joinpath(".vscode-server").absolute()
logger: logging.Logger = logging.getLogger(__name__)


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
        self.dev_container_config_path = (
            Path(config_name)
            if config_name.startswith("/")
            else Path.cwd()
            .joinpath(
                f"{config_name}",
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

    async def parse_all_extensions(self) -> list[str]:
        # use json5 for parsing json files that may contain comments
        parsed_config: dict[str, Any] = await asyncio.to_thread(
            lambda: json5.loads(self.dev_container_config_path.read_text())
        )
        extensions: list[str] = list(
            parsed_config.get("customizations", {})
            .get("vscode", {})
            .get("extensions", [])
        )

        for extension_id in extensions:
            metadata = await self.api_manager.get_extension_metadata(extension_id)
            # only store the results for the latest version
            versions = metadata.get(extension_id, [])
            if not versions:
                continue
            metadata = versions[0]
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
            with tempfile.TemporaryDirectory(prefix="vscode-extension.") as tmp_dir:
                file_path = Path(tmp_dir, self.get_filename(extension_id))

                with open(file_path, "wb") as f:
                    logger.info(f"Installing {extension_id} from {url}")
                    response: requests.Response = self.api_manager.session.get(
                        url, stream=True, headers=headers
                    )
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())

                target_path = self.target_path.joinpath(self.get_filename(extension_id))
                shutil.move(file_path, target_path)
                return target_path

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
            with tempfile.TemporaryDirectory(
                prefix="vscode-extension-signature."
            ) as tmp_dir:
                file_path = Path(tmp_dir, self.get_signature_filename(extension_id))

                with open(file_path, "wb") as f:
                    response: requests.Response = self.api_manager.session.get(
                        url, stream=True, headers=headers
                    )
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())

                target_path = self.target_path.joinpath(
                    self.get_signature_filename(extension_id)
                )
                shutil.move(file_path, target_path)
                return target_path

        return await asyncio.to_thread(_download_sync)

    async def verify_extension_signature(
        self, extension_path: Path, signature_archive_path: Path
    ) -> None:
        """Verify one VSIX against its signature archive using vsce-sign."""
        if self.vsce_sign_binary is None:
            raise RuntimeError("vsce-sign binary is not initialized for this run")

        cmd = [
            str(self.vsce_sign_binary),
            "verify",
            "--package",
            str(extension_path),
            "--signaturearchive",
            str(signature_archive_path),
        ]
        process = await asyncio.to_thread(
            subprocess.run,
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
        self, extension_id: str = "", extension_ids: list[str] = []
    ):
        self.installed = await self.find_installed()

        if extension_id:
            extension_ids.append(extension_id)

        for eid in extension_ids:
            # TODO: parse and update contents!
            installation_metadata = self.extension_metadata.get(eid, {}).get(
                "installation_metadata"
            )
            if installation_metadata:
                self.installed.append(installation_metadata)

        def _update_sync() -> None:
            json_path: Path = self.extensions_json_path()
            backup_path: Path = json_path.rename(f"{json_path}.bak")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.installed, f)
            except Exception:
                if backup_path.is_file():
                    shutil.move(backup_path, json_path)
                if json_path.stat().st_size == 0:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump("[]", f)

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
                cmd = [
                    self.code_binary,
                    "--install-extension",
                    f"{file_path}",
                    "--force",
                ]
                output = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    check=True,
                    text=True,
                )
                error_msg = "Error: "
                if error_msg in f"{output.stdout}" or error_msg in f"{output.stderr}":
                    raise subprocess.CalledProcessError(
                        1, cmd, output.stdout, output.stderr
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

    asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser while keeping existing install option names."""
    parser = argparse.ArgumentParser(prog="uvscem")
    parser.add_argument(
        "command",
        nargs="?",
        default="install",
        choices=["install"],
    )
    parser.add_argument("--config-name", default="devcontainer.json")
    parser.add_argument("--code-path", default="code")
    parser.add_argument("--target-path", default="$HOME/cache/.vscode/extensions")
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    install(
        config_name=args.config_name,
        code_path=args.code_path,
        target_path=args.target_path,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
