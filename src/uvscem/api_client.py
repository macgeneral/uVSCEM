#! /bin/env python3
import datetime
import logging
from pathlib import Path

# for parsing devcontainer.json (if it includes comments etc.)
import requests
import typer
from requests.adapters import HTTPAdapter, Retry

__author__ = "Arne Fahrenwalde"


# attempt to install an extension a maximum of three times
max_retries = 3
user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15"
# VSCode extension installation directory
vscode_root: Path = Path.home().joinpath(".vscode-server").absolute()
app: typer.Typer = typer.Typer()
logger: logging.Logger = logging.getLogger(__name__)


class CodeAPIManager(object):
    """Directly obtain all relevant information from the VSCode Marketplace API."""

    session: requests.Session | None = None

    def get_vscode_extension(
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
        api_version="7.2-preview.1",
    ):
        headers = {
            "Accept": f"application/json; charset=utf-8; api-version={api_version}"
        }

        flags = 0
        if include_versions:
            flags |= 0x1

        if include_files:
            flags |= 0x2

        if include_category_and_tags:
            flags |= 0x4

        if include_shared_accounts:
            flags |= 0x8

        if include_shared_accounts:
            flags |= 0x8

        if include_version_properties:
            flags |= 0x10

        if exclude_non_validated:
            flags |= 0x20

        if include_installation_targets:
            flags |= 0x40

        if include_asset_uri:
            flags |= 0x80

        if include_statistics:
            flags |= 0x100

        if include_latest_version_only:
            flags |= 0x200

        if unpublished:
            flags |= 0x1000

        if include_name_conflict_info:
            flags |= 0x8000

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
            body = {
                "filters": [
                    {
                        "criteria": [
                            {"filterType": 8, "value": "Microsoft.VisualStudio.Code"},
                            {"filterType": 7, "value": extension_id},
                        ],
                        "pageNumber": page,
                        "pageSize": page_size,
                        "sortBy": 0,
                        "sortOrder": 2,
                    }
                ],
                "assetTypes": [],
                "flags": flags,
            }

            r = self.session.post(
                "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            response = r.json()

            extensions = response["results"][0]["extensions"]
            for extension in extensions:
                yield extension

            if len(extensions) != page_size:
                break

    def get_extension_metadata(
        self,
        extension_id: str,
        include_latest_version_only: bool = False,
        include_latest_stable_version_only: bool = True,
    ) -> dict:
        extensions = {}
        logger.info(f"Obtaining metadata for {extension_id}")
        for extension in self.get_vscode_extension(
            extension_id=extension_id,
            include_latest_version_only=include_latest_version_only,
        ):
            result: list = []
            extension_name: str = extension.get("extensionName")
            extensions_versions: list = extension.get("versions")
            extensions_statistics: dict = dict(
                {
                    (item.get("statisticName"), item.get("value"))
                    for item in extension["statistics"]
                }
            )
            extension_publisher_username: str = extension.get("publisher").get(
                "publisherName"
            )

            for extension_version_info in extensions_versions:
                extension_version: str = extension_version_info.get("version")
                extension_properties: dict = dict(
                    {
                        (item.get("key"), item.get("value"))
                        for item in extension_version_info.get("properties", {})
                    }
                )
                extension_dependencies: list = [
                    x
                    for x in extension_properties.get(
                        "Microsoft.VisualStudio.Code.ExtensionDependencies"
                    ).split(",")
                    if x
                ]
                extension_packs: list = [
                    x
                    for x in extension_properties.get(
                        "Microsoft.VisualStudio.Code.ExtensionPack"
                    ).split(",")
                    if x
                ]
                extension_files: dict = dict(
                    {
                        (item.get("assetType"), item.get("source"))
                        for item in extension_version_info.get("files")
                    }
                )
                extension_download_url: str = extension_files.get(
                    "Microsoft.VisualStudio.Services.VSIXPackage"
                )
                is_pre_release: bool = extension_properties.get(
                    "Microsoft.VisualStudio.Code.PreRelease", False
                )
                # skip pre-release versions
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
                                        datetime.datetime.now(datetime.UTC)
                                        - datetime.datetime(
                                            1970, 1, 1, tzinfo=datetime.UTC
                                        )
                                    ).total_seconds()
                                    * 1000
                                ),
                                "id": extension["extensionId"],
                                "publisherDisplayName": extension["publisher"][
                                    "displayName"
                                ],
                                "publisherId": extension["publisher"]["publisherId"],
                                "isPreReleaseVersion": is_pre_release,
                            },
                        },
                    }
                )
                logger.debug(f"- {extension_download_url}")

                if include_latest_stable_version_only:
                    break
            extensions[extension_id] = result
        return extensions

    def __init__(self):
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
