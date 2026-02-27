from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Protocol
from urllib.parse import quote

import requests

DEFAULT_VSCE_SIGN_VERSION = "2.0.6"
PACKAGE_PREFIX = "@vscode/vsce-sign-"
SUPPORTED_VSCE_SIGN_TARGETS = (
    "linux-x64",
    "linux-arm64",
    "linux-arm",
    "alpine-x64",
    "alpine-arm64",
    "darwin-x64",
    "darwin-arm64",
    "win32-x64",
    "win32-arm64",
)


class VsceSignBootstrapError(RuntimeError):
    """Raised when vsce-sign binary bootstrap fails."""


class RegistrySession(Protocol):
    def get(self, url: str, *, timeout: int) -> requests.Response: ...


@dataclass
class VsceSignRunInstallation:
    """Represents one vsce-sign installation used for a full program run."""

    binary_path: Path
    temporary_dir: Path | None = None

    def cleanup(self) -> None:
        """Remove temporary install directory, if this run used one."""
        if self.temporary_dir and self.temporary_dir.exists():
            shutil.rmtree(self.temporary_dir, ignore_errors=True)


def _is_musl_linux() -> bool:
    libc_name, _ = platform.libc_ver()
    if libc_name.lower() == "musl":
        return True
    return Path("/etc/alpine-release").is_file()


def get_vsce_sign_target() -> str:
    """Return the vsce-sign platform target string used by npm packages."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "alpine-x64" if _is_musl_linux() else "linux-x64"
        if machine in {"aarch64", "arm64"}:
            return "alpine-arm64" if _is_musl_linux() else "linux-arm64"
        if machine in {"armv7l", "armv6l", "arm"}:
            return "linux-arm"
        raise VsceSignBootstrapError(f"Unsupported Linux architecture: {machine}")

    if system == "darwin":
        if machine in {"x86_64", "amd64"}:
            return "darwin-x64"
        if machine in {"arm64", "aarch64"}:
            return "darwin-arm64"
        raise VsceSignBootstrapError(f"Unsupported macOS architecture: {machine}")

    if system == "windows":
        if machine in {"x86_64", "amd64"}:
            return "win32-x64"
        if machine in {"arm64", "aarch64"}:
            return "win32-arm64"
        raise VsceSignBootstrapError(f"Unsupported Windows architecture: {machine}")

    raise VsceSignBootstrapError(f"Unsupported operating system: {system}")


def get_vsce_sign_package_name(target: str | None = None) -> str:
    """Return the platform-specific npm package name for vsce-sign."""
    resolved_target = target if target is not None else get_vsce_sign_target()
    return f"{PACKAGE_PREFIX}{resolved_target}"


def _verify_npm_integrity(tarball_bytes: bytes, integrity: str) -> None:
    if not integrity.startswith("sha512-"):
        raise VsceSignBootstrapError(
            f"Unsupported npm integrity algorithm: {integrity}"
        )

    expected = integrity.split("-", maxsplit=1)[1]
    actual = base64.b64encode(hashlib.sha512(tarball_bytes).digest()).decode("ascii")
    if expected != actual:
        raise VsceSignBootstrapError(
            "Integrity verification failed while downloading vsce-sign binary package"
        )


def _fetch_package_info(
    package_name: str,
    version: str,
    session: RegistrySession,
) -> tuple[str, str]:
    encoded_name = quote(package_name, safe="")
    response = session.get(f"https://registry.npmjs.org/{encoded_name}", timeout=30)
    response.raise_for_status()

    metadata = response.json()
    version_data = metadata.get("versions", {}).get(version)
    if not version_data:
        raise VsceSignBootstrapError(
            f"Version {version} not found for package {package_name}"
        )

    dist = version_data.get("dist", {})
    tarball_url = str(dist.get("tarball", ""))
    integrity = str(dist.get("integrity", ""))
    if not tarball_url or not integrity:
        raise VsceSignBootstrapError(
            f"Missing tarball metadata for package {package_name}@{version}"
        )
    return tarball_url, integrity


def _binary_name_for_target(target: str) -> str:
    return "vsce-sign.exe" if target.startswith("win32-") else "vsce-sign"


def _binary_name() -> str:
    return _binary_name_for_target(get_vsce_sign_target())


def _extract_binary_bytes_from_tarball(
    tarball_bytes: bytes,
    binary_name: str,
) -> bytes:
    archive_member = f"package/bin/{binary_name}"
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as archive:
        try:
            member = archive.getmember(archive_member)
        except KeyError as exc:
            raise VsceSignBootstrapError(
                f"Expected binary {archive_member} not found in package archive"
            ) from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise VsceSignBootstrapError(
                f"Expected binary {archive_member} not found in package archive"
            )
        return extracted.read()


def _expected_binary_sha512(
    package_name: str,
    version: str,
    binary_name: str,
    session: RegistrySession,
) -> str:
    tarball_url, integrity = _fetch_package_info(package_name, version, session)
    tarball_response = session.get(tarball_url, timeout=60)
    tarball_response.raise_for_status()
    tarball_bytes = tarball_response.content
    _verify_npm_integrity(tarball_bytes, integrity)
    expected_bytes = _extract_binary_bytes_from_tarball(tarball_bytes, binary_name)
    return hashlib.sha512(expected_bytes).hexdigest()


def install_vsce_sign_binary(
    install_dir: str | Path,
    version: str = DEFAULT_VSCE_SIGN_VERSION,
    force: bool = False,
    session: RegistrySession | None = None,
    verify_existing_checksum: bool = True,
) -> Path:
    """Install platform-specific vsce-sign binary without requiring Node.js."""
    target = get_vsce_sign_target()
    return install_vsce_sign_binary_for_target(
        target=target,
        install_dir=install_dir,
        version=version,
        force=force,
        session=session,
        verify_existing_checksum=verify_existing_checksum,
    )


def install_vsce_sign_binary_for_target(
    target: str,
    install_dir: str | Path,
    version: str = DEFAULT_VSCE_SIGN_VERSION,
    force: bool = False,
    session: RegistrySession | None = None,
    verify_existing_checksum: bool = True,
) -> Path:
    """Install vsce-sign binary for a specific npm platform target."""
    package_name = get_vsce_sign_package_name(target)
    install_path = Path(install_dir).expanduser().resolve()
    binary_name = _binary_name_for_target(target)
    binary_path = install_path.joinpath(binary_name)
    session_client: RegistrySession = (
        session if session is not None else requests.Session()
    )

    if binary_path.is_file() and not force:
        if not verify_existing_checksum:
            return binary_path
        expected_digest = _expected_binary_sha512(
            package_name=package_name,
            version=version,
            binary_name=binary_name,
            session=session_client,
        )
        actual_digest = hashlib.sha512(binary_path.read_bytes()).hexdigest()
        if actual_digest == expected_digest:
            return binary_path
        force = True

    install_path.mkdir(parents=True, exist_ok=True)
    tarball_url, integrity = _fetch_package_info(package_name, version, session_client)
    tarball_response = session_client.get(tarball_url, timeout=60)
    tarball_response.raise_for_status()
    tarball_bytes = tarball_response.content
    _verify_npm_integrity(tarball_bytes, integrity)
    binary_bytes = _extract_binary_bytes_from_tarball(tarball_bytes, binary_name)

    with NamedTemporaryFile(delete=False, dir=install_path) as tmp_file:
        tmp_target = Path(tmp_file.name)

    try:
        with open(tmp_target, "wb") as output:
            output.write(binary_bytes)

        if binary_name == "vsce-sign":
            current_mode = os.stat(tmp_target).st_mode
            os.chmod(
                tmp_target,
                current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
        tmp_target.replace(binary_path)
    finally:
        if tmp_target.exists() and tmp_target != binary_path:
            tmp_target.unlink(missing_ok=True)

    return binary_path


def install_vsce_sign_binary_with_fallback(
    install_dir: str | Path,
    version: str = DEFAULT_VSCE_SIGN_VERSION,
    force: bool = False,
    session: RegistrySession | None = None,
    temp_root: str | Path | None = None,
    verify_existing_checksum: bool = True,
) -> VsceSignRunInstallation:
    """Install vsce-sign with fallback to UUID temp dir when primary path is not writable."""
    try:
        binary_path = install_vsce_sign_binary(
            install_dir=install_dir,
            version=version,
            force=force,
            session=session,
            verify_existing_checksum=verify_existing_checksum,
        )
        return VsceSignRunInstallation(binary_path=binary_path)
    except PermissionError:
        pass

    base_temp_dir = (
        Path(temp_root).expanduser().resolve()
        if temp_root is not None
        else Path(tempfile.gettempdir()).resolve()
    )
    fallback_dir = base_temp_dir.joinpath(f"uvscem-vsce-sign-{uuid.uuid4()}")
    binary_path = install_vsce_sign_binary(
        install_dir=fallback_dir,
        version=version,
        force=True,
        session=session,
        verify_existing_checksum=False,
    )
    return VsceSignRunInstallation(binary_path=binary_path, temporary_dir=fallback_dir)


@contextmanager
def provision_vsce_sign_binary_for_run(
    install_dir: str | Path,
    version: str = DEFAULT_VSCE_SIGN_VERSION,
    force: bool = False,
    session: RegistrySession | None = None,
    temp_root: str | Path | None = None,
    verify_existing_checksum: bool = True,
) -> Iterator[Path]:
    """Provision one vsce-sign binary for a full run and always clean up temp fallback."""
    installation = install_vsce_sign_binary_with_fallback(
        install_dir=install_dir,
        version=version,
        force=force,
        session=session,
        temp_root=temp_root,
        verify_existing_checksum=verify_existing_checksum,
    )
    try:
        yield installation.binary_path
    finally:
        installation.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uvscem-vsce-sign-bootstrap")
    parser.add_argument("--install-dir", default="~/.local/bin")
    parser.add_argument("--version", default=DEFAULT_VSCE_SIGN_VERSION)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = install_vsce_sign_binary(
        install_dir=args.install_dir,
        version=args.version,
        force=args.force,
    )
    print(target)


if __name__ == "__main__":
    main()
