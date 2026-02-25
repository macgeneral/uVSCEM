from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow integration tests",
    )
    parser.addoption(
        "--only-slow",
        action="store_true",
        default=False,
        help="run only tests marked as slow",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    only_slow = bool(config.getoption("--only-slow"))
    run_slow = bool(config.getoption("--slow")) or only_slow

    if only_slow:
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            if "slow" in item.keywords:
                selected.append(item)
            else:
                deselected.append(item)

        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    if run_slow:
        return

    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
