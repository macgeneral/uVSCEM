# uVSCEM

[![PyPI - Version](https://img.shields.io/pypi/v/uvscem)](https://pypi.org/project/uvscem/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uvscem)](https://pypi.org/project/uvscem/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/uvscem)](https://pypi.org/project/uvscem/)
[![PyPI - License](https://img.shields.io/pypi/l/uvscem)](LICENSE)  
[![Lint](https://github.com/macgeneral/uVSCEM/actions/workflows/lint.yml/badge.svg)](https://github.com/macgeneral/uVSCEM/actions/workflows/lint.yml)
[![Test](https://github.com/macgeneral/uVSCEM/actions/workflows/test.yml/badge.svg)](https://github.com/macgeneral/uVSCEM/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/macgeneral/uVSCEM/branch/main/graph/badge.svg)](https://codecov.io/gh/macgeneral/uVSCEM)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

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

## Platform support

uVSCEM auto-detects common VS Code runtime environments and can fall back to local CLI discovery on macOS and Windows.

The primary tested target is still Linux, especially DevContainer and VS Code Remote (`.vscode-server`) environments.

## Install

### Pre-built binaries (since v1.0.4)

Self-contained binaries compiled with [Nuitka](https://nuitka.net) with no Python requirement are available on the [Releases](https://github.com/macgeneral/uVSCEM/releases) page for the following platforms:

| Platform        | File                    |
|-----------------|-------------------------|
| Linux x64       | `uvscem-linux-x64`      |
| Linux arm64     | `uvscem-linux-arm64`    |
| macOS arm64     | `uvscem-macos-arm64`    |
| Windows x64     | `uvscem-windows-x64`    |

Download the binary for your platform, make it executable, and run it directly:

```bash
# Linux / macOS
chmod +x ./uvscem
./uvscem install --config-name ./devcontainer.json
```

> **macOS note:** The binary is not code-signed. To remove the quarantine flag after downloading:
> ```bash
> xattr -d com.apple.quarantine ./uvscem
> ```

### Python package

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

Pinning is supported in `devcontainer.json` using `publisher.extension@version`:

```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint@3.0.10"
      ]
    }
  }
}
```

Export an offline bundle:

```bash
uvscem export --config-name ./devcontainer.json --bundle-path ./uvscem-offline-bundle
```

By default the bundle includes the `vsce-sign` binary for the current platform only. To bundle binaries for all platforms (useful when sharing the bundle across machines):

```bash
uvscem export --config-name ./devcontainer.json --vsce-sign-targets all
```

Or specify individual targets:

```bash
uvscem export --config-name ./devcontainer.json --vsce-sign-targets linux-x64,linux-arm64,darwin-arm64,win32-x64
```

Supported `--vsce-sign-targets` values: `current` (default), `all`, or a comma-separated list of `linux-x64`, `linux-arm64`, `linux-arm`, `alpine-x64`, `alpine-arm64`, `darwin-x64`, `darwin-arm64`, `win32-x64`, `win32-arm64`.

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

## A big thank you to the following people

- [Jossef Harush Kadouri](http://jossef.com/) for [this GitHub Gist](https://gist.github.com/jossef/8d7681ac0c7fd28e93147aa5044bc129) on how to query the undocumented VisualStudio Code Marketplace API, which I used as blueprint for [`marketplace.py`](https://github.com/macgeneral/uVSCEM/blob/main/src/uvscem/marketplace.py).
- [Ian McKellar](https://ianloic.com) for his blog post ["VSCode Remote and the command line"](https://ianloic.com/2021/02/16/vscode-remote-and-the-command-line/)  (notable mention: Lazy Ren@Stackoverflow for [this answer](https://stackoverflow.com/a/67916473) pointing me in this direction).
- [Michael Petrov](http://michaelpetrov.com) for [this answer](https://stackoverflow.com/a/62277798) on StackOverflow on how to test if a socket is closed in python.
