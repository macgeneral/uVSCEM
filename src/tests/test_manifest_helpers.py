from __future__ import annotations

from pathlib import Path

import pytest

from uvscem import bundle_io, marketplace


def test_bundle_io_coercion_helpers() -> None:
    assert bundle_io._as_string_list("invalid") == []
    assert bundle_io._as_string_list(["one", 2, None]) == ["one"]
    assert bundle_io._as_map_list("invalid") == []
    assert bundle_io._as_map_list([{"a": 1}, 2, "x"]) == [{"a": 1}]


def test_marketplace_map_list_helper_filters_non_dict_items() -> None:
    assert marketplace._as_map_list("invalid") == []
    assert marketplace._as_map_list([{"k": "v"}, 1, None]) == [{"k": "v"}]


def test_build_extension_manifest_entry_coerces_non_list_fields(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    (artifacts_dir / "pub.ext-1.0.0.vsix").write_bytes(b"vsix")
    (artifacts_dir / "pub.ext-1.0.0.sigzip").write_bytes(b"sig")

    entry = bundle_io.build_extension_manifest_entry(
        extension_id="pub.ext",
        metadata={
            "version": "1.0.0",
            "dependencies": "invalid",
            "extension_pack": 42,
        },
        artifacts_dir=artifacts_dir,
    )

    assert entry["dependencies"] == []
    assert entry["extension_pack"] == []


def test_resolve_bundled_vsce_sign_binary_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # A binary relative path that tries to escape the bundle directory.
    malicious_vsce_sign_info = {
        "binaries": [
            {"target": "linux-x64", "binary": "../../etc/passwd", "sha256": ""}
        ]
    }
    with pytest.raises(ValueError, match="escapes bundle directory"):
        bundle_io.resolve_bundled_vsce_sign_binary(
            bundle_dir=bundle_dir,
            vsce_sign_info=malicious_vsce_sign_info,
            current_target="linux-x64",
        )


def test_resolve_bundled_vsce_sign_binary_legacy_path_rejects_traversal(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Legacy single-binary path that tries to escape the bundle directory.
    malicious_vsce_sign_info = {"binary": "../outside/vsce-sign"}
    with pytest.raises(ValueError, match="escapes bundle directory"):
        bundle_io.resolve_bundled_vsce_sign_binary(
            bundle_dir=bundle_dir,
            vsce_sign_info=malicious_vsce_sign_info,
            current_target="linux-x64",
        )


def test_resolve_bundled_vsce_sign_binary_accepts_valid_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "vsce-sign" / "linux-x64").mkdir(parents=True)
    (bundle_dir / "vsce-sign" / "linux-x64" / "vsce-sign").write_bytes(b"bin")

    vsce_sign_info = {
        "binaries": [
            {
                "target": "linux-x64",
                "binary": "vsce-sign/linux-x64/vsce-sign",
                "sha256": "abc",
            }
        ]
    }
    binary_path, sha256 = bundle_io.resolve_bundled_vsce_sign_binary(
        bundle_dir=bundle_dir,
        vsce_sign_info=vsce_sign_info,
        current_target="linux-x64",
    )
    assert binary_path == bundle_dir / "vsce-sign" / "linux-x64" / "vsce-sign"
    assert sha256 == "abc"
