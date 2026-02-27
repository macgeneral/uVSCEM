"""Offline bundle export and import workflows.

Separated from extension_manager to keep that module focused on the
per-extension install lifecycle (download → verify → extract → register).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests

from uvscem.bundle_io import (
    build_bundle_manifest,
    build_extension_manifest_entry,
    resolve_bundled_vsce_sign_binary,
    resolve_vsce_sign_targets,
    sha256_file,
)
from uvscem.exceptions import (
    OfflineBundleExportError,
    OfflineBundleImportExecutionError,
    OfflineBundleImportMissingFileError,
    OfflineBundleImportValidationError,
    OfflineModeError,
)
from uvscem.vsce_sign_bootstrap import (
    DEFAULT_VSCE_SIGN_VERSION,
    SUPPORTED_VSCE_SIGN_TARGETS,
    get_vsce_sign_package_name,
    get_vsce_sign_target,
    install_vsce_sign_binary_for_target,
)

logger: logging.Logger = logging.getLogger(__name__)


def _workflow_error_message(operation: str, exc: Exception) -> str:
    return f"{operation} failed: {exc}"


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
    target_path: str = "",
    code_path: str = "code",
    log_level: str = "info",
    vsce_sign_version: str = DEFAULT_VSCE_SIGN_VERSION,
    vsce_sign_targets: str = "current",
    manifest_signing_key: str = "",
    allow_untrusted_urls: bool = False,
    allow_http: bool = False,
    disable_ssl_verification: bool = False,
    ca_bundle: str = "",
) -> None:
    """Export extensions, signatures, and vsce-sign into an offline bundle."""
    # Import here to avoid a circular dependency (bundle_workflow ← extension_manager).
    from uvscem.extension_manager import CodeExtensionManager

    _log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(_log_level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")
    logging.basicConfig(
        level=_log_level,
        format="%(relativeCreated)d [%(levelname)s] %(message)s",
    )

    config_path = Path(config_name)
    resolved_config_path = (
        config_path
        if config_path.is_absolute()
        else Path.cwd().joinpath(config_path).absolute()
    )
    if not resolved_config_path.is_file():
        raise OfflineBundleExportError(
            f"DevContainer configuration not found: {resolved_config_path}"
        )

    try:
        bundle_dir = Path(bundle_path).expanduser().resolve()
        artifacts_dir = bundle_dir.joinpath("artifacts")
        cache_dir = (
            Path.home().joinpath("cache/.vscode/extensions").resolve()
            if not target_path
            else Path(os.path.expandvars(target_path)).expanduser().resolve()
        )
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
            dict[str, dict[str, object]],
            dict[str, tuple[Path, Path]],
            requests.Session | None,
        ]:
            manager = CodeExtensionManager(
                config_name=config_name,
                code_path=code_path,
                target_directory=str(cache_dir),
                allow_unsigned=False,
                allow_untrusted_urls=allow_untrusted_urls,
                allow_http=allow_http,
                disable_ssl_verification=disable_ssl_verification,
                ca_bundle=ca_bundle,
            )
            extensions = await manager.parse_all_extensions()
            downloaded: dict[str, tuple[Path, Path]] = {}
            for extension_id in extensions:
                vsix_path = await manager.download_extension(extension_id)
                signature_path = await manager.download_signature_archive(extension_id)
                downloaded[extension_id] = (vsix_path, signature_path)
            return (
                extensions,
                manager.extension_metadata,
                downloaded,
                getattr(getattr(manager, "api_manager", None), "session", None),
            )

        extensions, extension_metadata, downloaded_artifacts, session_client = (
            asyncio.run(_run())
        )

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
                session=session_client,
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
    target_path: str = "",
    log_level: str = "info",
    strict_offline: bool = False,
    verify_manifest_signature: bool = True,
    manifest_verification_keyring: str = "",
) -> None:
    """Install extensions from a previously exported offline bundle."""
    from uvscem.extension_manager import CodeExtensionManager

    _log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(_log_level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")
    logging.basicConfig(
        level=_log_level,
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

                session.request = _offline_request

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
            artifacts_dir_resolved = artifacts_dir.resolve()
            try:
                source_vsix.resolve().relative_to(artifacts_dir_resolved)
                source_signature.resolve().relative_to(artifacts_dir_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Artifact path for {extension_id} escapes artifacts directory"
                ) from exc
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


__all__ = [
    "export_offline_bundle",
    "import_offline_bundle",
]
