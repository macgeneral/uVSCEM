from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtensionSpec:
    extension_id: str
    pinned_version: str = ""


@dataclass(frozen=True)
class ResolvedExtensionRequest:
    extension_id: str
    pinned_version: str = ""
