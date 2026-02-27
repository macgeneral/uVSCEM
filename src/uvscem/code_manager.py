#! /bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import socket
import tempfile
from pathlib import Path
from typing import TypedDict

from uvscem.vscode_paths import detect_runtime_environment, resolve_vscode_root

# for parsing devcontainer.json (if it includes comments etc.)
__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"

# VSCode extension installation directory
vscode_root: Path = resolve_vscode_root()
logger: logging.Logger = logging.getLogger(__name__)


class RemoteCliVersion(TypedDict):
    version: str
    commit: str
    version_tuple: tuple[int, ...]


def _parse_version_tuple(version: str) -> tuple[int, ...] | None:
    parts = version.strip().split(".")
    parsed_parts: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        parsed_parts.append(int(part))
    return tuple(parsed_parts)


def _parse_remote_cli_metadata(launcher_path: Path) -> tuple[str, str] | None:
    version: str = ""
    commit: str = ""
    with open(launcher_path, "r", encoding="utf-8") as launcher:
        for line in launcher:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", maxsplit=1)
            key = key.strip()
            if key not in {"VERSION", "COMMIT"}:
                continue
            value = raw_value.strip().strip('"').strip("'")
            if key == "VERSION":
                version = value
                continue
            commit = value

    if not version or not commit:
        return None
    return version, commit


class CodeManager(object):
    """Find VSCode CLI command and socket required for using 'code' in postAttachCommand."""

    socket_path: Path | None = None
    code_path: Path | None = None

    def __init__(self) -> None:
        self.socket_path = None
        self.code_path = None

    def _find_socket_sync(self, update_environment: bool = False) -> None:
        """Find all VSCode Unix sockets."""
        existing_hook = os.environ.get("VSCODE_IPC_HOOK_CLI", "").strip()
        if existing_hook:
            self.socket_path = Path(existing_hook)
            if update_environment:
                os.environ["VSCODE_IPC_HOOK_CLI"] = existing_hook
            return

        if platform.system().lower() == "windows":
            return

        # VSCode uses either /run/userid/ or /tmp/ if not set
        socket_dir: Path = Path(
            os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
        )
        sockets: list[Path] = sorted(socket_dir.glob("vscode-ipc-*.sock"))
        no_socket: bool = True

        for socket_path in sockets:
            if self.is_socket_closed(socket_path):
                logger.debug(f"Removing stale socket {socket_path}")
                Path.unlink(socket_path)
            else:
                logger.debug(f"Found active socket: {socket_path}")
                if no_socket:
                    self.socket_path = socket_path
                    no_socket = False

        if update_environment and self.socket_path is not None:
            os.environ["VSCODE_IPC_HOOK_CLI"] = f"{self.socket_path}"

    def _find_latest_code_sync(self, update_environment: bool = False) -> None:
        """Find all 'code' executables."""
        runtime_environment = detect_runtime_environment()
        vscode_versions: list[RemoteCliVersion] = []
        vscode_dir: Path = vscode_root.joinpath("bin")
        executables: list[Path] = []

        if runtime_environment in {"vscode-server", "vscode-remote"}:
            executables = list(vscode_dir.glob("*/bin/remote-cli/code"))

        for vsc in executables:
            metadata = _parse_remote_cli_metadata(vsc)
            if metadata is None:
                continue
            version, commit = metadata
            version_tuple = _parse_version_tuple(version)
            if version_tuple is None:
                continue
            vscode_versions.append(
                {
                    "version": version,
                    "commit": commit,
                    "version_tuple": version_tuple,
                }
            )

        if vscode_versions:
            latest_version = max(
                vscode_versions, key=lambda item: item["version_tuple"]
            )
            self.code_path = (
                vscode_dir.joinpath(f"{latest_version.get('commit')}")
                .resolve()
                .joinpath("bin/remote-cli/")
            )
        else:
            code_binary = shutil.which("code") or shutil.which("code-insiders")
            if not code_binary:
                logger.warning("No VSCode CLI executable found")
                return
            self.code_path = Path(code_binary).resolve().parent

        if update_environment:
            logger.debug(f"Adding Code [{self.code_path}] to $PATH")
            current_path = os.environ.get("PATH", "")
            vscode_path = f"{self.code_path}"
            if vscode_path in current_path:
                reordered_path = current_path.split(os.pathsep)
                # seems like the PATH is a bit messed up in VSCode, remove all duplicate entries
                no_duplicates = list(dict.fromkeys(reordered_path))
                no_duplicates.remove(vscode_path)
                # place the code cli as first entry
                no_duplicates.insert(0, vscode_path)
                os.environ["PATH"] = os.pathsep.join(no_duplicates)
            else:
                os.environ["PATH"] = (
                    f"{vscode_path}{os.pathsep}{current_path}"
                    if current_path
                    else vscode_path
                )

    @staticmethod
    def is_socket_closed(socket_path: Path) -> bool:
        """Detect if the socket is still usable because VSCode doesn't clean up old sockets."""
        if not all(
            hasattr(socket, attr) for attr in ("AF_UNIX", "MSG_DONTWAIT", "MSG_PEEK")
        ):
            return False

        client: socket.socket | None = None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Connect to the server
            client.connect(f"{socket_path}")
            # this will try to read bytes without blocking and also without removing them from buffer (peek only)
            client.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
        except BlockingIOError:
            return False  # socket is open and reading from it would block
        except ConnectionResetError:
            return True  # socket was closed for some other reason
        except ConnectionRefusedError:
            return True  # socket is dead
        except Exception:
            logger.exception("unexpected exception when checking if a socket is closed")
            return False
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        return False

    async def find_socket(self, update_environment: bool = False) -> None:
        """Asynchronously find VSCode Unix sockets."""
        await asyncio.to_thread(self._find_socket_sync, update_environment)

    async def find_latest_code(self, update_environment: bool = False) -> None:
        """Asynchronously find the latest VSCode remote CLI executable."""
        await asyncio.to_thread(self._find_latest_code_sync, update_environment)

    async def initialize(self) -> None:
        """Asynchronously initialize socket and CLI discovery."""
        await self.find_socket(update_environment=True)
        await self.find_latest_code(update_environment=True)
