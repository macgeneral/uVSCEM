from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

FLAG_INCLUDE_VERSIONS = 0x1
FLAG_INCLUDE_FILES = 0x2
FLAG_INCLUDE_CATEGORY_AND_TAGS = 0x4
FLAG_INCLUDE_SHARED_ACCOUNTS = 0x8
FLAG_INCLUDE_VERSION_PROPERTIES = 0x10
FLAG_EXCLUDE_NON_VALIDATED = 0x20
FLAG_INCLUDE_INSTALLATION_TARGETS = 0x40
FLAG_INCLUDE_ASSET_URI = 0x80
FLAG_INCLUDE_STATISTICS = 0x100
FLAG_INCLUDE_LATEST_VERSION_ONLY = 0x200
FLAG_UNPUBLISHED = 0x1000
FLAG_INCLUDE_NAME_CONFLICT_INFO = 0x8000


def build_query_flags(
    include_versions: bool,
    include_files: bool,
    include_category_and_tags: bool,
    include_shared_accounts: bool,
    include_version_properties: bool,
    exclude_non_validated: bool,
    include_installation_targets: bool,
    include_asset_uri: bool,
    include_statistics: bool,
    include_latest_version_only: bool,
    unpublished: bool,
    include_name_conflict_info: bool,
) -> int:
    flag_map = [
        (include_versions, FLAG_INCLUDE_VERSIONS),
        (include_files, FLAG_INCLUDE_FILES),
        (include_category_and_tags, FLAG_INCLUDE_CATEGORY_AND_TAGS),
        (include_shared_accounts, FLAG_INCLUDE_SHARED_ACCOUNTS),
        (include_version_properties, FLAG_INCLUDE_VERSION_PROPERTIES),
        (exclude_non_validated, FLAG_EXCLUDE_NON_VALIDATED),
        (include_installation_targets, FLAG_INCLUDE_INSTALLATION_TARGETS),
        (include_asset_uri, FLAG_INCLUDE_ASSET_URI),
        (include_statistics, FLAG_INCLUDE_STATISTICS),
        (include_latest_version_only, FLAG_INCLUDE_LATEST_VERSION_ONLY),
        (unpublished, FLAG_UNPUBLISHED),
        (include_name_conflict_info, FLAG_INCLUDE_NAME_CONFLICT_INFO),
    ]
    return sum(flag for enabled, flag in flag_map if enabled)


def build_extension_query_body(
    extension_id: str,
    page_number: int,
    page_size: int,
    flags: int,
) -> dict[str, Any]:
    return {
        "filters": [
            {
                "criteria": [
                    {"filterType": 8, "value": "Microsoft.VisualStudio.Code"},
                    {"filterType": 7, "value": extension_id},
                ],
                "pageNumber": page_number,
                "pageSize": page_size,
                "sortBy": 0,
                "sortOrder": 2,
            }
        ],
        "assetTypes": [],
        "flags": flags,
    }


def shape_extension_metadata_versions(
    extension: dict[str, Any],
    extension_id: str,
    include_latest_stable_version_only: bool,
    requested_version: str,
    vscode_root: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    extension_name: str = str(extension.get("extensionName", ""))
    extensions_versions: list[dict[str, Any]] = list(extension.get("versions", []))
    extensions_statistics: dict[str, Any] = dict(
        {
            (item.get("statisticName"), item.get("value"))
            for item in extension.get("statistics", [])
        }
    )
    extension_publisher_username: str = str(
        extension.get("publisher", {}).get("publisherName", "")
    )

    for extension_version_info in extensions_versions:
        extension_version: str = str(extension_version_info.get("version", ""))
        if requested_version and extension_version != requested_version:
            continue
        extension_properties: dict[str, Any] = dict(
            {
                (item.get("key"), item.get("value"))
                for item in extension_version_info.get("properties", {})
            }
        )
        extension_dependencies: list[str] = [
            item
            for item in str(
                extension_properties.get(
                    "Microsoft.VisualStudio.Code.ExtensionDependencies", ""
                )
            ).split(",")
            if item
        ]
        extension_packs: list[str] = [
            item
            for item in str(
                extension_properties.get(
                    "Microsoft.VisualStudio.Code.ExtensionPack", ""
                )
            ).split(",")
            if item
        ]
        extension_files: dict[str, Any] = dict(
            {
                (item.get("assetType"), item.get("source"))
                for item in extension_version_info.get("files", [])
            }
        )
        extension_download_url: str = str(
            extension_files.get("Microsoft.VisualStudio.Services.VSIXPackage", "")
        )
        extension_signature: str = str(
            extension_files.get("Microsoft.VisualStudio.Services.VsixSignature", "")
        )
        is_pre_release: bool = bool(
            extension_properties.get("Microsoft.VisualStudio.Code.PreRelease", False)
        )

        if is_pre_release and include_latest_stable_version_only:
            continue

        extension_path = vscode_root.joinpath(
            f"extensions/{extension_id}-{extension_version}"
        )
        result.append(
            {
                "publisher": extension_publisher_username,
                "name": extension_name,
                "version": extension_version,
                "dependencies": extension_dependencies,
                "extension_pack": extension_packs,
                "url": extension_download_url,
                "signature": extension_signature,
                "statistics": extensions_statistics,
                "installation_metadata": {
                    "identifier": {"id": extension_id},
                    "version": extension_version,
                    "location": {
                        "$mid": 1,
                        "path": f"{extension_path}",
                        "scheme": "file",
                    },
                    "relativeLocation": f"{extension_id}-{extension_version}",
                    "metadata": {
                        "installedTimestamp": int(
                            (
                                datetime.datetime.now(datetime.timezone.utc)
                                - datetime.datetime(
                                    1970,
                                    1,
                                    1,
                                    tzinfo=datetime.timezone.utc,
                                )
                            ).total_seconds()
                            * 1000
                        ),
                        "id": extension.get("extensionId", ""),
                        "publisherDisplayName": extension.get("publisher", {}).get(
                            "displayName", ""
                        ),
                        "publisherId": extension.get("publisher", {}).get(
                            "publisherId", ""
                        ),
                        "isPreReleaseVersion": is_pre_release,
                    },
                },
            }
        )

        if include_latest_stable_version_only:
            break

    return result
