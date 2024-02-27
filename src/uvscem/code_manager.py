#! /bin/env python3
from __future__ import annotations

import logging
import os
import shlex
import socket
import tempfile
from pathlib import Path

# for parsing devcontainer.json (if it includes comments etc.)
import typer

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"


# attempt to install an extension a maximum of three times
max_retries = 3
user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15"
# VSCode extension installation directory
vscode_root: Path = Path.home().joinpath(".vscode-server").absolute()
app: typer.Typer = typer.Typer()
logger: logging.Logger = logging.getLogger(__name__)


class CodeManager(object):
    """Find VSCode CLI command and socket required for using 'code' in postAttachCommand."""

    socket_path: Path | None = None
    code_path: Path | None = None

    def __init__(self) -> None:
        self.find_socket(update_environment=True)
        self.find_latest_code(update_environment=True)

    def find_socket(self, update_environment: bool = False) -> None:
        """Find all VSCode Unix sockets."""
        # VSCode uses either /run/userid/ or /tmp/ if not set
        socket_dir: Path = Path(
            os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
        )
        sockets: list = list(socket_dir.glob("vscode-ipc-*.sock"))
        no_socket: bool = True

        for s in sockets:
            socket_path = Path(socket_dir, s)
            if self.is_socket_closed(socket_path):
                logger.debug(f"Removing stale socket {socket_path}")
                Path.unlink(socket_path)
            else:
                logger.debug(f"Found active socket: {socket_path}")
                if no_socket:
                    self.socket_path = socket_path
                    no_socket = False

        if update_environment:
            os.environ["VSCODE_IPC_HOOK_CLI"] = f"{self.socket_path}"

    def find_latest_code(self, update_environment: bool = False) -> None:
        """Find all 'code' executables."""
        vscode_versions: dict = {}
        vscode_dir: Path = vscode_root.joinpath("bin")
        executables: list = list(vscode_dir.glob("*/bin/remote-cli/code"))

        for vsc in executables:
            # try to parse the important values from the shell script directly
            with open(vsc, "r") as sh:
                code_vars = dict(
                    shlex.split(line, posix=True)[0].split("=", 1)
                    for line in sh
                    if "=" in line and not line.startswith("ROOT")
                )
                vscode_versions[code_vars.get("VERSION")] = {
                    "version": code_vars.get("VERSION"),
                    "commit": code_vars.get("COMMIT"),
                }

        latest_version = vscode_versions.get(max(vscode_versions.keys()))
        # VSCode uses symlinks to /vscode/
        self.code_path = (
            vscode_dir.joinpath(f"{latest_version.get('commit')}")
            .resolve()
            .joinpath("bin/remote-cli/")
        )

        if update_environment:
            logger.debug(
                f"Adding Code [{self.code_path}] to $PATH\n  - Version: {latest_version.get('version')} | Commit: {latest_version.get('commit')}"
            )
            current_path = os.environ.get("PATH")
            vscode_path = f"{self.code_path}"
            if vscode_path in current_path:
                reordered_path = current_path.split(":")
                # seems like the PATH is a bit messed up in VSCode, remove all duplicate entries
                no_duplicates = list(dict.fromkeys(reordered_path))
                no_duplicates.remove(vscode_path)
                # place the code cli as first entry
                no_duplicates.insert(0, vscode_path)
                os.environ["PATH"] = ":".join(no_duplicates)
            else:
                os.environ["PATH"] = f"{vscode_path}:{current_path}"

    @staticmethod
    def is_socket_closed(socket_path: Path) -> bool:
        """Detect if the socket is still usable because VSCode doesn't clean up old sockets."""
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
        return False
