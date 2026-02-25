# Contributing to uVSCEM

Thanks for contributing.

This guide explains how to set up a local development environment, run checks, and submit changes.

## Ways to contribute

- Report bugs and edge cases
- Improve docs and examples
- Add tests for real-world proxy and offline scenarios
- Submit bug fixes and small improvements

## Development setup

From the repository root:

```bash
uv sync --group dev
```

## Quality gate (required)

Run the full quality gate before opening a PR:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest src
```

## Running slow tests

By default, slow tests are skipped.

Run only slow tests:

```bash
uv run pytest src --slow --no-cov
```

## Project layout

- `src/uvscem/api_client.py`: Marketplace API calls and metadata shaping
- `src/uvscem/code_manager.py`: VS Code CLI/socket discovery
- `src/uvscem/extension_manager.py`: CLI commands and extension workflows
- `src/tests/`: unit and integration tests

## Coding expectations

- Keep changes minimal and focused
- Preserve sync-first behavior unless a change explicitly requires otherwise
- Reuse existing helpers before adding new code paths
- Keep tests deterministic where possible
- Do not leak secrets/tokens in logs or error messages

## Pull requests

Please include:

- What changed and why
- How you tested it
- Any behavior changes users should know about

Small, focused PRs are preferred.

## Release and CI notes

- CI workflows are in `.github/workflows/`
- Releases are tag-based and require signed tags (`vX.Y.Z`)
- `publish.yml` verifies tag signatures before publishing

Repository variables used by release verification:

- `PYPI_TAG_SIGNER_GITHUB_LOGIN` (required)
- `PYPI_TAG_SIGNER` (optional)

## Security and signing

uVSCEM verifies extension signatures with `vsce-sign` and supports signed offline manifests.

Useful commands:

```bash
uvscem export --config-name ./devcontainer.json --manifest-signing-key YOUR_GPG_KEY_ID
uvscem import --bundle-path ./uvscem-offline-bundle --verify-manifest-signature
```
