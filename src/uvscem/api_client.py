#! /bin/env python3
from __future__ import annotations

import asyncio
import logging
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter, Retry

from uvscem.internal_config import (
    HTTP_REQUEST_TIMEOUT_SECONDS,
    HTTP_RETRY_ALLOWED_METHODS,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_STATUS_FORCELIST,
    HTTP_RETRY_TOTAL,
)
from uvscem.marketplace import (
    build_extension_query_body,
    build_query_flags,
    shape_extension_metadata_versions,
)
from uvscem.vscode_paths import resolve_vscode_root

__author__ = "Arne Fahrenwalde <arne@fahrenwal.de>"

logger: logging.Logger = logging.getLogger(__name__)

JsonMap = dict[str, object]


class CodeAPIManager:
    """Directly obtain all relevant information from the VSCode Marketplace API."""

    session: requests.Session

    def __init__(self) -> None:
        retry_strategy = Retry(
            total=HTTP_RETRY_TOTAL,
            backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
            status_forcelist=HTTP_RETRY_STATUS_FORCELIST,
            allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_vscode_extension_sync(
        self,
        extension_id: str = "",
        max_page: int = 100,
        page_size: int = 10,
        include_versions: bool = True,
        include_files: bool = True,
        include_category_and_tags: bool = True,
        include_shared_accounts: bool = True,
        include_version_properties: bool = True,
        exclude_non_validated: bool = False,
        include_installation_targets: bool = True,
        include_asset_uri: bool = True,
        include_statistics: bool = True,
        include_latest_version_only: bool = False,
        unpublished: bool = False,
        include_name_conflict_info: bool = True,
        api_version: str = "7.2-preview.1",
    ) -> Iterator[JsonMap]:
        headers = {
            "Accept": f"application/json; charset=utf-8; api-version={api_version}"
        }

        flags = build_query_flags(
            include_versions=include_versions,
            include_files=include_files,
            include_category_and_tags=include_category_and_tags,
            include_shared_accounts=include_shared_accounts,
            include_version_properties=include_version_properties,
            exclude_non_validated=exclude_non_validated,
            include_installation_targets=include_installation_targets,
            include_asset_uri=include_asset_uri,
            include_statistics=include_statistics,
            include_latest_version_only=include_latest_version_only,
            unpublished=unpublished,
            include_name_conflict_info=include_name_conflict_info,
        )

        for page in range(1, max_page + 1):
            """
            filterType:
                Tag = 1,
                ExtensionId = 4,
                Category = 5,
                ExtensionName = 7,
                Target = 8,
                Featured = 9,
                SearchText = 10,
                ExcludeWithFlags = 12

            SortBy:
                NoneOrRelevance = 0,
                LastUpdatedDate = 1,
                Title = 2,
                PublisherName = 3,
                InstallCount = 4,
                PublishedDate = 10,
                AverageRating = 6,
                WeightedRating = 12

            SortOrder:
                Default = 0,
                Ascending = 1,
                Descending = 2
            """
            body = build_extension_query_body(
                extension_id=extension_id,
                page_number=page,
                page_size=page_size,
                flags=flags,
            )

            r = self.session.post(
                "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery",
                json=body,
                headers=headers,
                timeout=HTTP_REQUEST_TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            response = r.json()
            if not isinstance(response, dict):
                break
            results = response.get("results", [])
            if not isinstance(results, list) or not results:
                break
            first_result = results[0]
            if not isinstance(first_result, dict):
                break
            extensions_value = first_result.get("extensions", [])
            if not isinstance(extensions_value, list):
                break
            extensions = [
                extension
                for extension in extensions_value
                if isinstance(extension, dict)
            ]
            for extension in extensions:
                yield extension

            if len(extensions) != page_size:
                break

    def _get_extension_metadata_sync(
        self,
        extension_id: str,
        include_latest_version_only: bool = False,
        include_latest_stable_version_only: bool = True,
        requested_version: str = "",
    ) -> dict[str, list[JsonMap]]:
        extensions: dict[str, list[JsonMap]] = {}
        logger.info(
            "Obtaining metadata for %s",
            extension_id.replace("\n", "\\n").replace("\r", "\\r"),
        )
        for extension in self._get_vscode_extension_sync(
            extension_id=extension_id,
            include_latest_version_only=include_latest_version_only,
        ):
            result = shape_extension_metadata_versions(
                extension=extension,
                extension_id=extension_id,
                include_latest_stable_version_only=include_latest_stable_version_only,
                requested_version=requested_version,
                vscode_root=resolve_vscode_root(),
            )
            for item in result:
                logger.debug("- %s", item.get("url", ""))
            extensions[extension_id] = result
        return extensions

    async def get_vscode_extension(
        self,
        extension_id: str = "",
        max_page: int = 100,
        page_size: int = 10,
        include_versions: bool = True,
        include_files: bool = True,
        include_category_and_tags: bool = True,
        include_shared_accounts: bool = True,
        include_version_properties: bool = True,
        exclude_non_validated: bool = False,
        include_installation_targets: bool = True,
        include_asset_uri: bool = True,
        include_statistics: bool = True,
        include_latest_version_only: bool = False,
        unpublished: bool = False,
        include_name_conflict_info: bool = True,
        api_version: str = "7.2-preview.1",
    ) -> list[JsonMap]:
        """Asynchronously fetch extension search results."""
        return await asyncio.to_thread(
            lambda: list(
                self._get_vscode_extension_sync(
                    extension_id=extension_id,
                    max_page=max_page,
                    page_size=page_size,
                    include_versions=include_versions,
                    include_files=include_files,
                    include_category_and_tags=include_category_and_tags,
                    include_shared_accounts=include_shared_accounts,
                    include_version_properties=include_version_properties,
                    exclude_non_validated=exclude_non_validated,
                    include_installation_targets=include_installation_targets,
                    include_asset_uri=include_asset_uri,
                    include_statistics=include_statistics,
                    include_latest_version_only=include_latest_version_only,
                    unpublished=unpublished,
                    include_name_conflict_info=include_name_conflict_info,
                    api_version=api_version,
                )
            )
        )

    async def get_extension_metadata(
        self,
        extension_id: str,
        include_latest_version_only: bool = False,
        include_latest_stable_version_only: bool = True,
        requested_version: str = "",
    ) -> dict[str, list[JsonMap]]:
        """Asynchronously fetch metadata for one extension."""
        return await asyncio.to_thread(
            self._get_extension_metadata_sync,
            extension_id,
            include_latest_version_only,
            include_latest_stable_version_only,
            requested_version,
        )
