from __future__ import annotations

from pathlib import Path

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
