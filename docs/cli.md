# uVSCEM CLI Reference

uVSCEM is invoked as `uvscem <command> [options]`.

```
uvscem [--version] <command> [options]
```

## Global options

| Option | Description |
|---|---|
| `--version` | Print the installed uvscem version and User-Agent string, then exit. |

---

## `install`

Download and install all extensions listed in a `devcontainer.json`.

```bash
uvscem install --config-name ./devcontainer.json
```

This is the default command — `uvscem --config-name ./devcontainer.json` is equivalent.

### How it works

1. Reads `customizations.vscode.extensions` from the config file (supports JSON5/comments).
2. Fetches extension metadata and dependency graphs from the VS Code Marketplace API.
3. Resolves a topological install order so dependencies are installed before dependents.
4. Downloads each VSIX and its marketplace signature archive.
5. Verifies each VSIX via `vsce-sign` (automatically bootstrapped on first run).
6. Installs via the `code --install-extension` CLI, falling back to direct VSIX extraction when the CLI is unavailable.
7. Skips extensions that are already installed at the correct version.

### Options

| Option | Default | Description |
|---|---|---|
| `--config-name PATH` | `devcontainer.json` | Path to the `devcontainer.json` to read extensions from. Relative paths are resolved from the current working directory. |
| `--code-path PATH` | `code` | Path to the VS Code CLI binary. Useful when `code` is not on `$PATH` or you need to target `code-insiders`. |
| `--target-path PATH` | `~/cache/.vscode/extensions` | Directory where downloaded VSIX and signature files are cached before installation. |
| `--allow-unsigned` | off | Allow installation of extensions that have no marketplace signature metadata. **INSECURE** — only use when an extension is known to be unsigned and trusted by other means. |
| `--allow-untrusted-urls` | off | Allow downloading from hosts outside the trusted marketplace allowlist (`marketplace.visualstudio.com`, `*.vsassets.io`, etc.). **INSECURE** — required only for non-standard mirrors. |
| `--allow-http` | off | Allow `http://` download URLs in addition to `https://`. **INSECURE** — intended for environments with SSL-terminating proxies that rewrite URLs. |
| `--disable-ssl-verification` | off | Disable TLS certificate verification for all HTTP requests in this run. **INSECURE** — prefer `--ca-bundle` for corporate CAs instead. |
| `--ca-bundle PATH` | (system trust store) | Path to a PEM CA bundle file to use instead of the system trust store. Use this to trust a corporate root CA without disabling verification entirely. |
| `--log-level LEVEL` | `info` | Logging verbosity. Accepted values: `debug`, `info`, `warning`, `error`, `critical`. |

### Pinning extension versions

Append `@version` to any extension ID in `devcontainer.json`:

```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint@3.0.10",
        "ms-python.python"
      ]
    }
  }
}
```

Pinned extensions are fetched at the exact requested version. Unpinned extensions always resolve to the latest available stable release.

---

## `export`

Download extensions and package them into a portable offline bundle for later air-gapped import.

```bash
uvscem export --config-name ./devcontainer.json --bundle-path ./uvscem-offline-bundle
```

### How it works

1. Resolves and fetches the same extension set as `install` (including dependency ordering).
2. Downloads every VSIX and its signature archive.
3. Downloads the platform-specific `vsce-sign` binary for each requested platform target.
4. Writes a `manifest.json` that records extension metadata, SHA-256 checksums, ordered install list, and bundled binary paths.
5. Optionally signs `manifest.json` with GPG to allow the receiver to verify bundle authenticity.

The resulting bundle directory is self-contained and can be transferred to an air-gapped machine.

### Options

| Option | Default | Description |
|---|---|---|
| `--config-name PATH` | `devcontainer.json` | Path to the `devcontainer.json` to read extensions from. |
| `--code-path PATH` | `code` | VS Code CLI binary path (used for environment initialisation only during export). |
| `--target-path PATH` | `~/cache/.vscode/extensions` | Local cache directory for downloaded files. |
| `--bundle-path PATH` | `./uvscem-offline-bundle` | Destination directory for the offline bundle output. Created if it does not exist. |
| `--vsce-sign-version VERSION` | (current default) | The `vsce-sign` npm package version to bundle. Defaults to the version baked into this uvscem release. Change only if you need to target a specific verifier version. |
| `--vsce-sign-targets TARGETS` | `current` | Which platform targets to include vsce-sign binaries for. `current` includes only the platform running the export. `all` includes every supported platform. A comma-separated list selects specific platforms (e.g. `linux-x64,darwin-arm64,win32-x64`). Supported values: `linux-x64`, `linux-arm64`, `linux-arm`, `alpine-x64`, `alpine-arm64`, `darwin-x64`, `darwin-arm64`, `win32-x64`, `win32-arm64`. |
| `--manifest-signing-key KEY_ID` | (no signing) | A GPG key ID (or fingerprint) used to create a detached ASCII-armored signature (`manifest.json.asc`) alongside the manifest. The signing key must be available in your local GPG keyring. |
| `--allow-untrusted-urls` | off | See [install](#options). |
| `--allow-http` | off | See [install](#options). |
| `--disable-ssl-verification` | off | See [install](#options). |
| `--ca-bundle PATH` | (system trust store) | See [install](#options). |
| `--log-level LEVEL` | `info` | See [install](#options). |

### Multi-platform bundle example

```bash
# Include vsce-sign binaries for all platforms in one bundle
uvscem export \
  --config-name ./devcontainer.json \
  --bundle-path ./bundle \
  --vsce-sign-targets all

# Include only the platforms you need
uvscem export \
  --config-name ./devcontainer.json \
  --bundle-path ./bundle \
  --vsce-sign-targets linux-x64,linux-arm64,darwin-arm64
```

### Signing the bundle manifest

```bash
uvscem export \
  --config-name ./devcontainer.json \
  --bundle-path ./bundle \
  --manifest-signing-key YOUR_GPG_KEY_ID
```

This creates `bundle/manifest.json.asc`. Pass `--manifest-verification-keyring` on import to verify against a specific keyring (see below).

---

## `import`

Install extensions from a previously exported offline bundle without requiring network access.

```bash
uvscem import --bundle-path ./uvscem-offline-bundle
```

### How it works

1. Reads `manifest.json` from the bundle directory.
2. Optionally verifies the GPG detached signature (`manifest.json.asc`).
3. Validates SHA-256 checksums for every VSIX and signature archive listed in the manifest.
4. Copies artifacts into the local cache directory.
5. Selects the bundled `vsce-sign` binary matching the current platform.
6. Verifies and installs each extension in the dependency-ordered sequence from the manifest, skipping already-installed versions.

### Options

| Option | Default | Description |
|---|---|---|
| `--bundle-path PATH` | `./uvscem-offline-bundle` | Path to the bundle directory created by `export`. Must contain `manifest.json`. |
| `--code-path PATH` | `code` | VS Code CLI binary path, used the same way as in `install`. |
| `--target-path PATH` | `~/cache/.vscode/extensions` | Local cache directory where artifacts are staged before installation. |
| `--strict-offline` | off | Block any outbound network request during the import run. Useful for verifying that the bundle is truly self-contained and for enforcing air-gapped policies. If a network request is attempted (e.g. by a code path that tries to reach the Marketplace), the import will fail with a clear error. |
| `--skip-manifest-signature-verification` | off | Skip verification of `manifest.json.asc` even when the file exists. Use only with bundles you produced yourself in a secure pipeline where manifest signing was omitted. |
| `--manifest-verification-keyring PATH` | (default GPG keyring) | Path to a GPG keyring file to use when verifying `manifest.json.asc` instead of the default keyring. Useful in CI or restricted environments where only a specific public key should be trusted. |
| `--log-level LEVEL` | `info` | See [install](#options). |

### Signature verification behaviour

By default:

- If `manifest.json.asc` is present, `import` verifies it before reading any other bundle content.
- If `manifest.json.asc` is absent, the import proceeds without signature verification.

To require a signature even when the file might be absent, verify the `.asc` file's existence manually before invoking `import`.

To skip verification entirely (e.g. for a bundle you produced locally):

```bash
uvscem import \
  --bundle-path ./bundle \
  --skip-manifest-signature-verification
```

To verify against a specific public key only:

```bash
uvscem import \
  --bundle-path ./bundle \
  --manifest-verification-keyring /path/to/trusted.gpg
```

---

## Shared behaviour

### Security defaults

All three commands enforce the following by default. Each can be relaxed only via an explicit flag with a clearly insecure name.

| Behaviour | Default | Override flag |
|---|---|---|
| Extension signature verification | required | `--allow-unsigned` |
| HTTPS-only downloads | enforced | `--allow-http` |
| Trusted-host allowlist | enforced | `--allow-untrusted-urls` |
| TLS certificate verification | on | `--disable-ssl-verification` / `--ca-bundle` |

### Retry and fallback

`install` (and the install step of `import`) attempt `code --install-extension` first. On failure it retries up to 3 times, re-probing the VS Code IPC socket before each attempt. If all retries are exhausted, uVSCEM falls back to direct VSIX extraction — unpacking the `.vsix` into the extensions directory and updating `extensions.json` itself.

### Proxy support

All HTTP/HTTPS requests respect the standard proxy environment variables:

```
HTTP_PROXY, HTTPS_PROXY, NO_PROXY
```

Set them in your shell or container environment before running uVSCEM. No additional configuration is required.

### Environment variables

| Variable | Description |
|---|---|
| `UVSCEM_VSCODE_ROOT` | Override the VS Code data root (where `extensions/` and `extensions.json` live). Useful when auto-detection resolves the wrong path on macOS or Windows. |
| `UVSCEM_RUNTIME` | Override the detected runtime environment. Accepted values: `local`, `vscode-server`, `vscode-remote`. |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | Standard proxy variables respected by all HTTP requests. |
| `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` | Override the CA bundle used by the Requests library when `--ca-bundle` is not set. |
