from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Protocol

import requests


class DownloadSession(Protocol):
    def get(
        self,
        url: str,
        *,
        stream: bool,
        headers: dict[str, str],
        timeout: tuple[int, int],
    ) -> requests.Response: ...


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def stream_download_to_target(
    *,
    session: DownloadSession,
    url: str,
    target_path: Path,
    headers: dict[str, str],
    temp_prefix: str,
    timeout: tuple[int, int],
) -> Path:
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmp_dir:
        file_path = Path(tmp_dir, target_path.name)

        with open(file_path, "wb") as output:
            response: requests.Response = session.get(
                url,
                stream=True,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=1024 * 8):
                if chunk:
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        shutil.move(file_path, target_path)
        return target_path


def run_vsce_sign_verify(
    *,
    vsce_sign_binary: Path,
    extension_path: Path,
    signature_archive_path: Path,
    run_command: RunCommand = subprocess.run,
) -> None:
    cmd = [
        str(vsce_sign_binary),
        "verify",
        "--package",
        str(extension_path),
        "--signaturearchive",
        str(signature_archive_path),
    ]
    process = run_command(
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


def run_code_cli_install(
    *,
    code_binary: str,
    extension_path: Path,
    run_command: RunCommand = subprocess.run,
) -> None:
    cmd = [
        code_binary,
        "--install-extension",
        f"{extension_path}",
        "--force",
    ]
    output = run_command(
        cmd,
        capture_output=True,
        check=True,
        text=True,
    )
    error_msg = "Error: "
    if error_msg in f"{output.stdout}" or error_msg in f"{output.stderr}":
        raise subprocess.CalledProcessError(1, cmd, output.stdout, output.stderr)
