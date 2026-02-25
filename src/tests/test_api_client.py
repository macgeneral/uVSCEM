from __future__ import annotations

from types import SimpleNamespace

from uvscem.api_client import CodeAPIManager


def test_init_creates_session_with_http_adapters() -> None:
    manager = CodeAPIManager()

    assert "https://" in manager.session.adapters
    assert "http://" in manager.session.adapters


def test_get_vscode_extension_handles_pagination_and_flags() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    calls: list[dict] = []

    class _Response:
        def __init__(self, extensions: list[dict]) -> None:
            self._extensions = extensions

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"extensions": self._extensions}]}

    def _post(url: str, json: dict, headers: dict) -> _Response:
        calls.append({"url": url, "json": json, "headers": headers})
        if len(calls) == 1:
            return _Response([{"id": "one"}, {"id": "two"}])
        return _Response([{"id": "three"}])

    manager.session = SimpleNamespace(post=_post)

    result = list(
        manager.get_vscode_extension(
            extension_id="publisher.name",
            page_size=2,
            include_category_and_tags=False,
            include_asset_uri=False,
            include_latest_version_only=True,
            unpublished=True,
            include_name_conflict_info=True,
        )
    )

    assert [item["id"] for item in result] == ["one", "two", "three"]
    assert len(calls) == 2
    assert calls[0]["headers"]["Accept"].endswith("api-version=7.2-preview.1")
    assert calls[0]["json"]["filters"][0]["criteria"][1]["value"] == "publisher.name"
    assert calls[0]["json"]["flags"] == (
        0x1 | 0x2 | 0x8 | 0x10 | 0x40 | 0x100 | 0x200 | 0x1000 | 0x8000
    )


def _build_extension_payload() -> dict:
    return {
        "extensionId": "ext-guid",
        "extensionName": "sample-extension",
        "statistics": [{"statisticName": "install", "value": 42}],
        "publisher": {
            "publisherName": "sample-publisher",
            "displayName": "Sample Publisher",
            "publisherId": "publisher-guid",
        },
        "versions": [
            {
                "version": "2.0.0-pre",
                "properties": [
                    {
                        "key": "Microsoft.VisualStudio.Code.PreRelease",
                        "value": True,
                    },
                    {
                        "key": "Microsoft.VisualStudio.Code.ExtensionDependencies",
                        "value": "dep.one,dep.two",
                    },
                    {
                        "key": "Microsoft.VisualStudio.Code.ExtensionPack",
                        "value": "pack.one",
                    },
                ],
                "files": [
                    {
                        "assetType": "Microsoft.VisualStudio.Services.VSIXPackage",
                        "source": "https://download/pre.vsix",
                    },
                    {
                        "assetType": "Microsoft.VisualStudio.Services.VsixSignature",
                        "source": "https://download/pre.sig",
                    },
                ],
            },
            {
                "version": "1.9.0",
                "properties": [
                    {
                        "key": "Microsoft.VisualStudio.Code.PreRelease",
                        "value": False,
                    },
                    {
                        "key": "Microsoft.VisualStudio.Code.ExtensionDependencies",
                        "value": "dep.one,dep.two",
                    },
                    {
                        "key": "Microsoft.VisualStudio.Code.ExtensionPack",
                        "value": "pack.one",
                    },
                ],
                "files": [
                    {
                        "assetType": "Microsoft.VisualStudio.Services.VSIXPackage",
                        "source": "https://download/stable.vsix",
                    },
                    {
                        "assetType": "Microsoft.VisualStudio.Services.VsixSignature",
                        "source": "https://download/stable.sig",
                    },
                ],
            },
        ],
    }


def test_get_extension_metadata_skips_prerelease_by_default() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager.get_vscode_extension = lambda **_kwargs: [_build_extension_payload()]

    result = manager.get_extension_metadata("publisher.name")
    versions = result["publisher.name"]

    assert len(versions) == 1
    metadata = versions[0]
    assert metadata["version"] == "1.9.0"
    assert metadata["dependencies"] == ["dep.one", "dep.two"]
    assert metadata["extension_pack"] == ["pack.one"]
    assert metadata["url"] == "https://download/stable.vsix"
    assert metadata["signature"] == "https://download/stable.sig"
    assert metadata["statistics"]["install"] == 42
    assert metadata["installation_metadata"]["identifier"]["id"] == "publisher.name"
    assert metadata["installation_metadata"]["metadata"]["isPreReleaseVersion"] is False


def test_get_extension_metadata_can_include_prerelease_versions() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager.get_vscode_extension = lambda **_kwargs: [_build_extension_payload()]

    result = manager.get_extension_metadata(
        "publisher.name", include_latest_stable_version_only=False
    )
    versions = result["publisher.name"]

    assert [item["version"] for item in versions] == ["2.0.0-pre", "1.9.0"]
    assert (
        versions[0]["installation_metadata"]["metadata"]["isPreReleaseVersion"] is True
    )
