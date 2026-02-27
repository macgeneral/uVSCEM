from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from uvscem.api_client import CodeAPIManager


def test_init_creates_session_with_http_adapters() -> None:
    manager = CodeAPIManager()

    assert "https://" in manager.session.adapters
    assert "http://" in manager.session.adapters
    retry = manager.session.adapters["https://"].max_retries
    assert "POST" in retry.allowed_methods


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

    def _post(url: str, json: dict, headers: dict, timeout: int) -> _Response:
        calls.append({"url": url, "json": json, "headers": headers})
        assert timeout == 30
        if len(calls) == 1:
            return _Response([{"id": "one"}, {"id": "two"}])
        return _Response([{"id": "three"}])

    manager.session = SimpleNamespace(post=_post)

    result = asyncio.run(
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


def test_get_vscode_extension_supports_minimal_flag_set() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    calls: list[dict] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"extensions": []}]}

    manager.session = SimpleNamespace(
        post=lambda url, json, headers, timeout: (
            calls.append({"url": url, "json": json, "headers": headers}) or _Response()
        )
    )

    result = asyncio.run(
        manager.get_vscode_extension(
            extension_id="publisher.name",
            max_page=1,
            page_size=10,
            include_versions=False,
            include_files=False,
            include_category_and_tags=False,
            include_shared_accounts=False,
            include_version_properties=False,
            exclude_non_validated=False,
            include_installation_targets=False,
            include_asset_uri=False,
            include_statistics=False,
            include_latest_version_only=False,
            unpublished=False,
            include_name_conflict_info=False,
            api_version="1.0-preview",
        )
    )

    assert result == []
    assert calls[0]["json"]["flags"] == 0
    assert calls[0]["headers"]["Accept"].endswith("api-version=1.0-preview")


def test_get_vscode_extension_covers_additional_flag_paths() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"extensions": []}]}

    manager.session = SimpleNamespace(post=lambda *_args, **_kwargs: _Response())

    result = asyncio.run(
        manager.get_vscode_extension(
            max_page=0,
            include_versions=False,
            include_files=False,
            include_category_and_tags=True,
            include_shared_accounts=False,
            include_version_properties=False,
            exclude_non_validated=True,
            include_installation_targets=False,
            include_asset_uri=True,
            include_statistics=False,
            include_latest_version_only=False,
            unpublished=False,
            include_name_conflict_info=False,
        )
    )

    assert result == []


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
    manager._get_vscode_extension_sync = lambda **_kwargs: [_build_extension_payload()]

    result = asyncio.run(manager.get_extension_metadata("publisher.name"))
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
    manager._get_vscode_extension_sync = lambda **_kwargs: [_build_extension_payload()]

    result = asyncio.run(
        manager.get_extension_metadata(
            "publisher.name", include_latest_stable_version_only=False
        )
    )
    versions = result["publisher.name"]

    assert [item["version"] for item in versions] == ["2.0.0-pre", "1.9.0"]
    assert (
        versions[0]["installation_metadata"]["metadata"]["isPreReleaseVersion"] is True
    )


def test_get_extension_metadata_handles_extension_without_versions() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager._get_vscode_extension_sync = lambda **_kwargs: [
        {
            "extensionId": "id",
            "extensionName": "name",
            "statistics": [],
            "publisher": {"publisherName": "publisher"},
            "versions": [],
        }
    ]

    result = asyncio.run(manager.get_extension_metadata("publisher.name"))

    assert result == {"publisher.name": []}


def test_get_extension_metadata_can_select_requested_version() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager._get_vscode_extension_sync = lambda **_kwargs: [_build_extension_payload()]

    result = asyncio.run(
        manager.get_extension_metadata(
            "publisher.name",
            include_latest_stable_version_only=False,
            requested_version="2.0.0-pre",
        )
    )
    versions = result["publisher.name"]

    assert [item["version"] for item in versions] == ["2.0.0-pre"]


def test_get_extension_metadata_requested_version_returns_empty_when_missing() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager._get_vscode_extension_sync = lambda **_kwargs: [_build_extension_payload()]

    result = asyncio.run(
        manager.get_extension_metadata(
            "publisher.name",
            include_latest_stable_version_only=False,
            requested_version="9.9.9",
        )
    )

    assert result == {"publisher.name": []}


def test_async_api_methods_use_private_sync_helpers() -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)
    manager._get_vscode_extension_sync = lambda **_kwargs: [
        {"id": "one"},
        {"id": "two"},
    ]
    manager._get_extension_metadata_sync = lambda *args, **kwargs: {
        "publisher.name": []
    }

    extensions = asyncio.run(
        manager.get_vscode_extension(extension_id="publisher.name")
    )
    metadata = asyncio.run(manager.get_extension_metadata("publisher.name"))

    assert extensions == [{"id": "one"}, {"id": "two"}]
    assert metadata == {"publisher.name": []}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"results": []},
        {"results": ["invalid"]},
        {"results": [{"extensions": "invalid"}]},
    ],
)
def test_get_vscode_extension_handles_malformed_payloads(payload: object) -> None:
    manager = CodeAPIManager.__new__(CodeAPIManager)

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return payload

    manager.session = SimpleNamespace(post=lambda *_args, **_kwargs: _Response())

    result = asyncio.run(
        manager.get_vscode_extension(extension_id="publisher.name", max_page=1)
    )

    assert result == []


def test_get_package_version_returns_zero_when_package_not_found() -> None:
    from uvscem.internal_config import _get_package_version

    assert _get_package_version("this-package-does-not-exist-xyz") == "0"


def test_get_uvscem_version_prefers_version_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from uvscem.internal_config import _get_uvscem_version

    version_module = types.ModuleType("uvscem._version")
    version_module.__version__ = "9.9.9"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvscem._version", version_module)

    assert _get_uvscem_version() == "9.9.9"
