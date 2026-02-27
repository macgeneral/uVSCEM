from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def _get_package_version(name: str) -> str:
    """Return the installed version of *name*, or ``"0"`` if not found."""
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "0"


def _get_uvscem_version() -> str:
    """Return the uvscem version.

    Prefers the ``uvscem._version`` module when present — the Nuitka build
    step writes that file into the installed package before compilation so
    the version string is a compile-time constant in the resulting binary.
    Falls back to ``importlib.metadata`` for normal installs and editable
    development environments.
    """
    import importlib

    try:
        mod = importlib.import_module("uvscem._version")
        return str(mod.__version__)
    except (ImportError, AttributeError):
        return _get_package_version("uvscem")


_uvscem_version = _get_uvscem_version()

# Note: some Marketplace CDN endpoints may return non-functional download URLs when
# presented with unfamiliar User-Agents. If downloads break, adjust this string.
DEFAULT_USER_AGENT = (
    f"uvscem/{_uvscem_version}"
    f" ({platform.system()}; {platform.machine()};"
    f" compatible; +https://github.com/macgeneral/uVSCEM)"
)

MAX_INSTALL_RETRIES = 3

HTTP_REQUEST_TIMEOUT_SECONDS = 30
HTTP_STREAM_CONNECT_TIMEOUT_SECONDS = 10
HTTP_STREAM_READ_TIMEOUT_SECONDS = 120

HTTP_RETRY_TOTAL = 3
HTTP_RETRY_BACKOFF_FACTOR = 1
HTTP_RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]
HTTP_RETRY_ALLOWED_METHODS = ["HEAD", "GET", "OPTIONS", "POST"]
