from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import cast

JsonMap = dict[str, object]


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_map_list(value: object) -> list[JsonMap]:
    if not isinstance(value, list):
        return []
    return [cast(JsonMap, item) for item in value if isinstance(item, dict)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_vsce_sign_targets(
    value: str,
    current_target: str,
    supported_targets: tuple[str, ...],
) -> list[str]:
    if value == "current":
        return [current_target]
    if value == "all":
        return list(supported_targets)
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("No valid vsce-sign targets were provided")
    return result


def build_extension_manifest_entry(
    extension_id: str,
    metadata: JsonMap,
    artifacts_dir: Path,
) -> JsonMap:
    version = str(metadata.get("version", ""))
    filename = f"{extension_id}-{version}.vsix"
    signature_filename = f"{extension_id}-{version}.sigzip"
    return {
        "id": extension_id,
        "version": version,
        "filename": filename,
        "signature_filename": signature_filename,
        "vsix_sha256": sha256_file(artifacts_dir.joinpath(filename)),
        "signature_sha256": sha256_file(artifacts_dir.joinpath(signature_filename)),
        "installation_metadata": metadata.get("installation_metadata", {}),
        "dependencies": _as_string_list(metadata.get("dependencies", [])),
        "extension_pack": _as_string_list(metadata.get("extension_pack", [])),
    }


def build_bundle_manifest(
    config_name: str,
    ordered_extensions: list[str],
    extension_entries: list[JsonMap],
    vsce_sign_version: str,
    vsce_sign_binaries: list[dict[str, str]],
) -> JsonMap:
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_name": config_name,
        "ordered_extensions": ordered_extensions,
        "extensions": extension_entries,
        "vsce_sign": {
            "version": vsce_sign_version,
            "binaries": vsce_sign_binaries,
        },
    }


def resolve_bundled_vsce_sign_binary(
    bundle_dir: Path,
    vsce_sign_info: JsonMap,
    current_target: str,
) -> tuple[Path, str]:
    binaries = _as_map_list(vsce_sign_info.get("binaries", []))
    if binaries:
        selected = None
        for item in binaries:
            if str(item.get("target", "")) == current_target:
                selected = item
                break
        if selected is None and len(binaries) == 1:
            selected = binaries[0]
        if selected is None:
            raise ValueError(
                f"No bundled vsce-sign binary found for target {current_target}"
            )

        binary_rel = str(selected.get("binary", ""))
        return bundle_dir.joinpath(binary_rel), str(selected.get("sha256", ""))

    binary_rel = str(vsce_sign_info.get("binary", ""))
    return bundle_dir.joinpath(binary_rel), ""
