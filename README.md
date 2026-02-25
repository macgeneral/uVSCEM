# uVSCEM

![PyPI - Version](https://img.shields.io/pypi/v/uvscem) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uvscem) ![PyPI - Implementation](https://img.shields.io/pypi/implementation/uvscem) ![PyPI - License](https://img.shields.io/pypi/l/uvscem) [![Lint](https://github.com/macgeneral/uVSCEM/actions/workflows/lint.yml/badge.svg)](https://github.com/macgeneral/uVSCEM/actions/workflows/lint.yml) [![Test](https://github.com/macgeneral/uVSCEM/actions/workflows/test.yml/badge.svg)](https://github.com/macgeneral/uVSCEM/actions/workflows/test.yml) [![Publish](https://github.com/macgeneral/uVSCEM/actions/workflows/publish.yml/badge.svg)](https://github.com/macgeneral/uVSCEM/actions/workflows/publish.yml)

uVSCEM is a VS Code extension installer for restricted environments.

It helps when normal extension installation fails because of proxy issues, controlled outbound access, or air-gapped workflows. It can also export and import offline extension bundles.

## What it does

- Reads extension IDs from your `devcontainer.json`.
- Downloads VSIX packages and marketplace signatures.
- Verifies extension signatures using `vsce-sign`.
- Installs extensions in a DevContainer-compatible way.
- Supports offline bundles (`export` and `import`).

## Requirements

- Python `3.10+`
- VS Code CLI (`code`) available in your environment
- Access to your `devcontainer.json`

## Install

```bash
pip install uvscem
```

You can also use `uv` or `pipx` if you prefer.

## Quick start (DevContainer)

Add uVSCEM to your container image, then call it from `postAttachCommand`:

```json
{
  "postAttachCommand": "uvscem install --config-name /path/to/devcontainer.json"
}
```

This installs (and updates) extensions listed in your config each time the container is attached.

## Commands

Install extensions directly:

```bash
uvscem install --config-name ./devcontainer.json
```

Export an offline bundle:

```bash
uvscem export --config-name ./devcontainer.json --bundle-path ./uvscem-offline-bundle
```

Import an offline bundle without network access:

```bash
uvscem import --bundle-path ./uvscem-offline-bundle --strict-offline
```

Optional manifest authenticity checks:

```bash
uvscem export --config-name ./devcontainer.json --manifest-signing-key YOUR_GPG_KEY_ID
uvscem import --bundle-path ./uvscem-offline-bundle --verify-manifest-signature
```

## Proxy note

uVSCEM follows standard proxy environment variables such as `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`.

## Why this exists

uVSCEM is a practical workaround for known VS Code proxy/devcontainer limitations, including:

- https://github.com/microsoft/vscode/issues/12588
- https://github.com/microsoft/vscode/issues/29910
- https://github.com/orgs/devcontainers/discussions/94

## Contributing

Development setup, testing, and release details are in [CONTRIBUTING.md](CONTRIBUTING.md).
