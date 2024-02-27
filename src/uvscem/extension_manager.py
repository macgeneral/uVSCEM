#! /bin/env python3
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# for parsing devcontainer.json (if it includes comments etc.)
import json5
import requests
import typer
from dependency_algorithm import Dependencies

from uvscem.api_client import CodeAPIManager
from uvscem.code_manager import CodeManager

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"


# attempt to install an extension a maximum of three times
max_retries = 3
user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15"
# VSCode extension installation directory
vscode_root: Path = Path.home().joinpath(".vscode-server").absolute()
app: typer.Typer = typer.Typer()
logger: logging.Logger = logging.getLogger(__name__)


class CodeExtensionManager(object):
    """Proxy aware helper script to install VSCode extensions in a DevContainer."""

    api_manager: CodeAPIManager | None = None
    socket_manager: CodeManager | None = None
    code_binary: str | None = None
    dev_container_config_path: Path | None = None
    extensions: list = []
    installed: dict = {}
    extension_dependencies: dict = defaultdict(set)
    extension_metadata: dict = {}
    target_path: Path | None = None

    def __init__(
        self,
        config_name: str = "devcontainer.json",
        code_path: str = "code",
        target_directory: str = "",
    ) -> None:
        """Set up the CodeExtensionManager."""
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
        self.extensions = self.parse_all_extensions()
        self.installed = self.find_installed()
        self.target_path = (
            Path(target_directory).absolute()
            if target_directory
            else Path.home().joinpath("cache/.vscode/extensions").absolute()
        )
        # create target directory if necessary
        self.target_path.mkdir(parents=True, exist_ok=True)

    def sanitize_dependencies(self, extensions: list) -> list:
        """
        Remove VSCode internal extensions which are not available
        on the Marketplace, but still are listed as dependency.
        """
        sanitized = []

        for e in extensions:
            if not e.startswith("vscode."):
                sanitized.append(e)

        return sanitized

    def add_missing_dependency(self, extensions: list, dependencies: list) -> None:
        for dependency in dependencies:
            if dependency not in extensions:
                extensions.append(dependency)

    def parse_all_extensions(self) -> list:
        # use json5 for parsing json files that may contain comments
        extensions: list = list(
            json5.loads(self.dev_container_config_path.read_text())
            .get("customizations")
            .get("vscode")
            .get("extensions")
        )

        for extension_id in extensions:
            metadata = self.api_manager.get_extension_metadata(extension_id)
            # only store the results for the latest version
            metadata = metadata.get(extension_id)[0]
            self.extension_metadata[extension_id] = metadata

            dependencies = self.sanitize_dependencies(metadata.get("dependencies"))
            self.extension_dependencies[extension_id].update(dependencies)
            self.add_missing_dependency(extensions, dependencies)

            # dont treat extension packs as dependencies
            extension_pack = self.sanitize_dependencies(metadata.get("extension_pack"))
            self.add_missing_dependency(extensions, extension_pack)

        # get installation order
        dependencies = Dependencies(self.extension_dependencies)
        return dependencies.resolve_dependencies()

    def install(self) -> None:
        """Install all extensions listed in devcontainer.json."""
        # don't attempt to download already installed extensions
        self.exclude_installed()

        while self.extensions:
            self.install_extension(self.extensions.pop(0))
            time.sleep(1)

    def get_dirname(self, extension_id: str) -> str:
        version: str = self.extension_metadata.get(extension_id, {}).get("version")
        return f"{extension_id}-{version}"

    def get_filename(self, extension_id: str) -> str:
        return f"{self.get_dirname(extension_id)}.vsix"

    def download_extension(self, extension_id: str) -> Path:
        """Download an extension and return the downloaded file's path."""
        metadata: dict = self.extension_metadata.get(extension_id, {})
        url: str = metadata.get("url")
        headers: dict = {
            "User-Agent": user_agent,
        }

        with tempfile.TemporaryDirectory(prefix="vscode-extension.") as tmp_dir:
            file_path = Path(tmp_dir, self.get_filename(extension_id))

            with open(file_path, "wb") as f:
                logger.info(f"Installing {extension_id} from {url}")
                response: requests.request = self.api_manager.session.get(
                    url, stream=True, headers=headers
                )
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())

            # move it to a cache directory after successful download
            target_path = self.target_path.joinpath(self.get_filename(extension_id))
            shutil.move(file_path, target_path)

            return target_path

    def install_extension_manually(
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

        # remove existing files
        if target_path.exists():
            shutil.rmtree(target_path)

        with tempfile.TemporaryDirectory(prefix="vscode-extension-extract.") as tmp_dir:
            with zipfile.ZipFile(extension_path, "r") as zip:
                zip.extractall(tmp_dir)
                shutil.move(Path(tmp_dir, "extension/"), target_path)

        # add metadata to extensions.json
        if update_json:
            self.update_extensions_json(extension_id=extension_id)

    def update_extensions_json(self, extension_id: str = "", extension_ids: list = []):
        self.installed = self.find_installed()

        if extension_id:
            extension_ids.append(extension_id)

        for eid in extension_ids:
            # TODO: parse and update contents!
            self.installed.append(
                self.extension_metadata.get(eid).get("installation_metadata")
            )

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

    def install_extension(
        self, extension_id: str, extension_pack: bool = False, retries: int = 0
    ) -> None:
        """Install a single extension."""
        file_name = self.get_filename(extension_id)
        file_path = self.target_path.joinpath(file_name)
        metadata = self.extension_metadata.get(extension_id)

        if file_path.is_file():
            logger.info(f"File {file_name} exists - skipping download...")
        else:
            file_path = self.download_extension(extension_id)

        # installing extension packs doesn't work because the code install routine tries fetching other packages itself
        if metadata.get("extension_pack"):
            manually_installed = []

            if not extension_pack:
                manually_installed.append(extension_id)
                extension_pack = True

            # also repeat this step for the packages listed in extension pack
            for ep in metadata.get("extension_pack"):
                self.install_extension(
                    self.extensions.pop(self.extensions.index(ep)), extension_pack=True
                )
                manually_installed.append(ep)

            self.update_extensions_json(extension_ids=manually_installed)

        if extension_pack:
            self.install_extension_manually(extension_id, file_path, update_json=False)
        else:
            try:
                # TODO: add a timeout if the code command hangs
                # TODO: add a fallback to manual installation after x retries
                # install it
                cmd = [
                    self.code_binary,
                    "--install-extension",
                    f"{file_path}",
                    "--force",
                ]
                output = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                )
                # catch some weird code errors
                error_msg = "Error: "
                if error_msg in f"{output.stdout}" or error_msg in f"{output.stderr}":
                    raise subprocess.CalledProcessError(
                        1, cmd, output.stdout, output.stderr
                    )
            except subprocess.CalledProcessError as e:
                logger.error(f"Something went wrong: {e}")
                self.socket_manager.find_socket(update_environment=True)
                if retries < max_retries:
                    retries += 1
                    self.install_extension(
                        extension_id, extension_pack=extension_pack, retries=retries
                    )

    def find_installed(self) -> dict:
        """Return a list of already installed extensions."""
        extensions_path: Path = self.extensions_json_path()
        extensions: dict = {}
        with open(extensions_path, "r") as f:
            extensions = json.load(f)

        return extensions

    def extensions_json_path(self) -> Path:
        return vscode_root.joinpath("extensions/extensions.json")

    def exclude_installed(self) -> None:
        """Remove all already installed extensions from the extensions list."""
        for installed_extension in self.installed:
            for extension in self.extensions:
                installed_version: str = installed_extension.get("version")
                installed_name: str = (
                    installed_extension.get("identifier").get("id").lower()
                )
                extension_name: str = extension.lower()
                extension_version: str = self.extension_metadata.get(extension).get(
                    "version"
                )
                if (
                    extension_name == installed_name
                    and installed_version == extension_version
                ):
                    self.extensions.remove(extension)
                    logger.info(
                        f"Skipping {extension} ({extension_version}), already installed."
                    )


@app.command()
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
    CodeExtensionManager(config_name=config_name, code_path=code_path).install()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
