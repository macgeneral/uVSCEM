from __future__ import annotations

import base64
import hashlib
import io
import os
import runpy
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from uvscem import vsce_sign_bootstrap
from uvscem.vsce_sign_bootstrap import VsceSignBootstrapError, VsceSignRunInstallation


class _Response:
    def __init__(self, payload=None, content: bytes = b"") -> None:
        self._payload = payload if payload is not None else {}
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Session:
    verify: bool | str = True

    def __init__(self, responses: dict[str, _Response]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: int):
        self.calls.append(url)
        response = self.responses.get(url)
        if response is None:
            raise AssertionError(f"unexpected url: {url}")
        return response


def _build_tar(member_name: str, payload: bytes) -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return data.getvalue()


def _sha512_integrity(blob: bytes) -> str:
    digest = hashlib.sha512(blob).digest()
    return f"sha512-{base64.b64encode(digest).decode('ascii')}"


def test_is_musl_linux_prefers_libc_ver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap.platform, "libc_ver", lambda: ("musl", "1.2")
    )
    monkeypatch.setattr(vsce_sign_bootstrap.Path, "is_file", lambda self: False)

    assert vsce_sign_bootstrap._is_musl_linux() is True


def test_is_musl_linux_uses_alpine_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap.platform, "libc_ver", lambda: ("glibc", "2.36")
    )

    def _is_file(self: Path) -> bool:
        return self == Path("/etc/alpine-release")

    monkeypatch.setattr(vsce_sign_bootstrap.Path, "is_file", _is_file)

    assert vsce_sign_bootstrap._is_musl_linux() is True


def test_is_musl_linux_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap.platform, "libc_ver", lambda: ("glibc", "2.36")
    )
    monkeypatch.setattr(vsce_sign_bootstrap.Path, "is_file", lambda self: False)

    assert vsce_sign_bootstrap._is_musl_linux() is False


@pytest.mark.parametrize(
    ("system", "machine", "musl", "expected"),
    [
        ("Linux", "x86_64", False, "linux-x64"),
        ("Linux", "amd64", True, "alpine-x64"),
        ("Linux", "aarch64", False, "linux-arm64"),
        ("Linux", "arm64", True, "alpine-arm64"),
        ("Linux", "armv7l", False, "linux-arm"),
        ("Darwin", "x86_64", False, "darwin-x64"),
        ("Darwin", "arm64", False, "darwin-arm64"),
        ("Windows", "amd64", False, "win32-x64"),
        ("Windows", "arm64", False, "win32-arm64"),
    ],
)
def test_get_vsce_sign_target(
    system: str,
    machine: str,
    musl: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: system)
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: machine)
    monkeypatch.setattr(vsce_sign_bootstrap, "_is_musl_linux", lambda: musl)

    assert vsce_sign_bootstrap.get_vsce_sign_target() == expected


def test_get_vsce_sign_target_unsupported_linux_arch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: "mips64")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.get_vsce_sign_target()


def test_get_vsce_sign_target_unsupported_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: "x86_64")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.get_vsce_sign_target()


def test_get_vsce_sign_target_unsupported_darwin_arch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: "ppc64")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.get_vsce_sign_target()


def test_get_vsce_sign_target_unsupported_windows_arch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Windows")
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: "x86")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.get_vsce_sign_target()


def test_get_vsce_sign_package_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap, "get_vsce_sign_target", lambda: "linux-x64"
    )

    assert (
        vsce_sign_bootstrap.get_vsce_sign_package_name()
        == "@vscode/vsce-sign-linux-x64"
    )


def test_get_vsce_sign_package_name_with_explicit_target() -> None:
    assert (
        vsce_sign_bootstrap.get_vsce_sign_package_name("win32-x64")
        == "@vscode/vsce-sign-win32-x64"
    )


def test_binary_name_uses_current_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_target",
        lambda: "win32-x64",
    )

    assert vsce_sign_bootstrap._binary_name() == "vsce-sign.exe"


def test_verify_npm_integrity_success() -> None:
    blob = b"payload"
    vsce_sign_bootstrap._verify_npm_integrity(blob, _sha512_integrity(blob))


def test_verify_npm_integrity_rejects_other_algorithms() -> None:
    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap._verify_npm_integrity(b"x", "sha256-abc")


def test_verify_npm_integrity_mismatch() -> None:
    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap._verify_npm_integrity(b"x", _sha512_integrity(b"y"))


def test_fetch_package_info_success() -> None:
    session = _Session(
        {
            "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64": _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": "https://example/pkg.tgz",
                                "integrity": "sha512-abc",
                            }
                        }
                    }
                }
            )
        }
    )

    tarball_url, integrity = vsce_sign_bootstrap._fetch_package_info(
        "@vscode/vsce-sign-linux-x64", "2.0.6", session
    )

    assert tarball_url == "https://example/pkg.tgz"
    assert integrity == "sha512-abc"


def test_fetch_package_info_missing_version() -> None:
    session = _Session(
        {
            "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64": _Response(
                {"versions": {}}
            )
        }
    )

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap._fetch_package_info(
            "@vscode/vsce-sign-linux-x64", "2.0.6", session
        )


def test_fetch_package_info_missing_dist_fields() -> None:
    session = _Session(
        {
            "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64": _Response(
                {"versions": {"2.0.6": {"dist": {}}}}
            )
        }
    )

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap._fetch_package_info(
            "@vscode/vsce-sign-linux-x64", "2.0.6", session
        )


def test_install_vsce_sign_binary_no_force_keeps_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "vsce-sign"
    existing.write_text("existing")

    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    class _NeverSession:
        verify: bool | str = True

        def get(self, *_args, **_kwargs):
            raise AssertionError("network should not be used")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary(
        install_dir=tmp_path,
        force=False,
        session=_NeverSession(),
        verify_existing_checksum=False,
    )

    assert installed == existing
    assert existing.read_text() == "existing"


def test_install_existing_binary_with_matching_checksum_keeps_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "vsce-sign"
    existing.write_bytes(b"binary")

    tar_payload = _build_tar("package/bin/vsce-sign", b"binary")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary(
        install_dir=tmp_path,
        version="2.0.6",
        force=False,
        session=session,
    )

    assert installed == existing
    assert existing.read_bytes() == b"binary"


def test_install_existing_binary_checksum_mismatch_reinstalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "vsce-sign"
    existing.write_bytes(b"old")

    tar_payload = _build_tar("package/bin/vsce-sign", b"new")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary(
        install_dir=tmp_path,
        version="2.0.6",
        force=False,
        session=session,
    )

    assert installed == existing
    assert existing.read_bytes() == b"new"


def test_install_vsce_sign_binary_installs_binary_and_sets_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tar_payload = _build_tar("package/bin/vsce-sign", b"binary")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary(
        install_dir=tmp_path,
        version="2.0.6",
        force=True,
        session=session,
    )

    assert installed == tmp_path / "vsce-sign"
    assert installed.read_bytes() == b"binary"
    assert os.access(installed, os.X_OK)


def test_install_vsce_sign_binary_missing_member_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tar_payload = _build_tar("package/bin/other", b"binary")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.install_vsce_sign_binary(
            install_dir=tmp_path,
            version="2.0.6",
            force=True,
            session=session,
        )


def test_install_vsce_sign_binary_windows_member_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tar_payload = _build_tar("package/bin/vsce-sign.exe", b"binary")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-win32-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-win32-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Windows")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary(
        install_dir=tmp_path,
        version="2.0.6",
        force=True,
        session=session,
    )

    assert installed.name == "vsce-sign.exe"
    assert installed.read_bytes() == b"binary"


def test_install_vsce_sign_binary_extractfile_none_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        info = tarfile.TarInfo(name="package/bin/vsce-sign")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    tar_payload = data.getvalue()

    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.install_vsce_sign_binary(
            install_dir=tmp_path,
            version="2.0.6",
            force=True,
            session=session,
        )


def test_install_with_fallback_returns_primary_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary" / "vsce-sign"

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary",
        lambda install_dir, version, force, session, verify_existing_checksum: primary,
    )

    installed = vsce_sign_bootstrap.install_vsce_sign_binary_with_fallback(
        install_dir=tmp_path / "target",
    )

    assert installed.binary_path == primary
    assert installed.temporary_dir is None


def test_install_with_fallback_uses_uuid_temp_dir_on_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    fallback_binary = tmp_path / "temp-root" / "uvscem-vsce-sign-fixed" / "vsce-sign"

    def _install(install_dir, version, force, session):
        install_path = Path(install_dir)
        calls.append(install_path)
        if len(calls) == 1:
            raise PermissionError("no write access")
        return fallback_binary

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary",
        lambda install_dir, version, force, session, verify_existing_checksum: _install(
            install_dir, version, force, session
        ),
    )
    monkeypatch.setattr(vsce_sign_bootstrap.uuid, "uuid4", lambda: "fixed")

    installed = vsce_sign_bootstrap.install_vsce_sign_binary_with_fallback(
        install_dir=tmp_path / "forbidden",
        temp_root=tmp_path / "temp-root",
    )

    assert calls[0] == tmp_path / "forbidden"
    assert calls[1] == tmp_path / "temp-root" / "uvscem-vsce-sign-fixed"
    assert installed.binary_path == fallback_binary
    assert installed.temporary_dir == tmp_path / "temp-root" / "uvscem-vsce-sign-fixed"


def test_install_with_fallback_uses_system_temp_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def _install(install_dir, version, force, session):
        install_path = Path(install_dir)
        calls.append(install_path)
        if len(calls) == 1:
            raise PermissionError("no write access")
        return install_path / "vsce-sign"

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary",
        lambda install_dir, version, force, session, verify_existing_checksum: _install(
            install_dir, version, force, session
        ),
    )
    monkeypatch.setattr(vsce_sign_bootstrap.uuid, "uuid4", lambda: "default-temp")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "sys-tmp"))

    installed = vsce_sign_bootstrap.install_vsce_sign_binary_with_fallback(
        install_dir=tmp_path / "forbidden",
    )

    assert calls[1] == tmp_path / "sys-tmp" / "uvscem-vsce-sign-default-temp"
    assert (
        installed.temporary_dir
        == tmp_path / "sys-tmp" / "uvscem-vsce-sign-default-temp"
    )


def test_install_with_fallback_does_not_swallow_non_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary",
        lambda install_dir, version, force, session, verify_existing_checksum: (
            _ for _ in ()
        ).throw(VsceSignBootstrapError("network failure")),
    )

    with pytest.raises(VsceSignBootstrapError):
        vsce_sign_bootstrap.install_vsce_sign_binary_with_fallback(
            install_dir=tmp_path / "target"
        )


def test_run_installation_cleanup_removes_temp_dir(tmp_path: Path) -> None:
    temp_dir = tmp_path / "tmp-install"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "vsce-sign").write_text("bin")

    installation = VsceSignRunInstallation(
        binary_path=temp_dir / "vsce-sign",
        temporary_dir=temp_dir,
    )
    installation.cleanup()

    assert not temp_dir.exists()


def test_run_installation_cleanup_noop_without_temp_dir(tmp_path: Path) -> None:
    binary = tmp_path / "vsce-sign"
    binary.write_text("bin")

    installation = VsceSignRunInstallation(binary_path=binary)
    installation.cleanup()

    assert binary.exists()


def test_install_vsce_sign_binary_unlinks_temp_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tar_payload = _build_tar("package/bin/vsce-sign", b"binary")
    package_url = "https://registry.npmjs.org/%40vscode%2Fvsce-sign-linux-x64"
    tarball_url = "https://example.com/pkg.tgz"
    session = _Session(
        {
            package_url: _Response(
                {
                    "versions": {
                        "2.0.6": {
                            "dist": {
                                "tarball": tarball_url,
                                "integrity": _sha512_integrity(tar_payload),
                            }
                        }
                    }
                }
            ),
            tarball_url: _Response(content=tar_payload),
        }
    )

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "get_vsce_sign_package_name",
        lambda target=None: "@vscode/vsce-sign-linux-x64",
    )
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")

    tmp_created = tmp_path / "tmp-created"

    class _Tmp:
        def __enter__(self):
            tmp_created.write_bytes(b"")
            return type("_TmpFile", (), {"name": str(tmp_created)})()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        vsce_sign_bootstrap, "NamedTemporaryFile", lambda **kwargs: _Tmp()
    )
    monkeypatch.setattr(
        Path,
        "replace",
        lambda self, target: (_ for _ in ()).throw(RuntimeError("replace failed")),
    )

    with pytest.raises(RuntimeError):
        vsce_sign_bootstrap.install_vsce_sign_binary(
            install_dir=tmp_path,
            version="2.0.6",
            force=True,
            session=session,
        )

    assert not tmp_created.exists()


def test_provision_for_run_cleans_temp_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = tmp_path / "temp-install"
    install_dir = tmp_path / "install-root"
    temp_dir.mkdir(parents=True, exist_ok=True)
    binary = temp_dir / "vsce-sign"
    binary.write_text("bin")

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary_with_fallback",
        lambda install_dir, version, force, session, temp_root, verify_existing_checksum: (
            VsceSignRunInstallation(
                binary_path=binary,
                temporary_dir=temp_dir,
            )
        ),
    )

    with vsce_sign_bootstrap.provision_vsce_sign_binary_for_run(
        str(install_dir)
    ) as path:
        assert path == binary
        assert temp_dir.exists()

    assert not temp_dir.exists()


def test_provision_for_run_keeps_temp_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = tmp_path / "temp-install"
    install_dir = tmp_path / "install-root"
    temp_dir.mkdir(parents=True, exist_ok=True)
    binary = temp_dir / "vsce-sign"
    binary.write_text("bin")

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary_with_fallback",
        lambda install_dir, version, force, session, temp_root, verify_existing_checksum: (
            VsceSignRunInstallation(
                binary_path=binary,
                temporary_dir=temp_dir,
            )
        ),
    )

    with pytest.raises(RuntimeError):
        with vsce_sign_bootstrap.provision_vsce_sign_binary_for_run(str(install_dir)):
            raise RuntimeError("run failed")

    assert not temp_dir.exists()


def test_build_parser_defaults() -> None:
    args = vsce_sign_bootstrap.build_parser().parse_args([])

    assert args.install_dir == "~/.local/bin"
    assert args.version == vsce_sign_bootstrap.DEFAULT_VSCE_SIGN_VERSION
    assert args.force is False


def test_main_prints_installed_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_path = Path(tempfile.gettempdir()) / "vsce-sign"
    install_dir = str(Path(tempfile.gettempdir()) / "bin")

    monkeypatch.setattr(
        vsce_sign_bootstrap,
        "install_vsce_sign_binary",
        lambda install_dir, version, force: expected_path,
    )
    monkeypatch.setattr(
        vsce_sign_bootstrap.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "_Args",
            (),
            {
                "install_dir": install_dir,
                "version": "2.0.6",
                "force": True,
            },
        )(),
    )

    vsce_sign_bootstrap.main()

    assert capsys.readouterr().out.strip() == str(expected_path)


def test_module_main_guard_executes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["uvscem-vsce-sign-bootstrap", "--help"])
    monkeypatch.delitem(sys.modules, "uvscem.vsce_sign_bootstrap", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("uvscem.vsce_sign_bootstrap", run_name="__main__")

    assert exc.value.code == 0


def test_install_vsce_sign_binary_for_target_uses_default_session_when_none_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers the else: session_client = requests.Session() path."""
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vsce_sign_bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(vsce_sign_bootstrap, "_is_musl_linux", lambda: False)

    binary_path = tmp_path / "vsce-sign"
    binary_path.write_bytes(b"fake-binary")

    # Binary present + verify_existing_checksum=False returns early without
    # touching the network, but session_client = requests.Session() is still hit.
    result = vsce_sign_bootstrap.install_vsce_sign_binary_for_target(
        target="linux-x64",
        install_dir=tmp_path,
        version="2.0.6",
        force=False,
        session=None,
        verify_existing_checksum=False,
    )

    assert result == binary_path
