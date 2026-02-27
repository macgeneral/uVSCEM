# Architecture Instruction File

Apply these rules when creating or modifying uVSCEM code.

## Keep module boundaries clear

- Package root is `src/uvscem`.
- CLI entrypoint is `uvscem.extension_manager:main` (see `pyproject.toml`).
- Keep responsibilities separated:
	- `api_client.py`: Marketplace HTTP calls + extension metadata shaping.
	- `code_manager.py`: VSCode socket/CLI discovery and environment setup.
	- `extension_manager.py`: config parsing, dependency ordering, install workflow.
- Avoid creating broad utility modules unless logic is reused in multiple places.

## Fit the current runtime model

- The project uses `asyncio` for all IO-bound work (HTTP requests, subprocess calls, filesystem operations); sync helpers are wrapped with `asyncio.to_thread` where needed.
- Do not introduce async for CPU-bound or purely in-memory logic.
- Preserve existing runtime assumptions for DevContainer/postAttach usage.

## Design principles

- Prefer small, composable functions over large multi-purpose routines.
- Favor straightforward implementations over clever abstractions.
- Keep APIs lean: add parameters/return data only when needed.
- Reuse existing helpers before adding new ones.
- Keep changes surgical; do not refactor unrelated modules.

## Domain-specific behavior to preserve

- Maintain proxy-aware extension download behavior.
- Keep compatibility with devcontainer extension lists in `customizations.vscode.extensions`.
- Preserve dependency and extension-pack handling order semantics.
- Preserve VSCode server path conventions (`~/.vscode-server/**`) unless explicitly changing behavior.

## Compatibility constraints

- Keep code compatible with Python 3.10+ (`requires-python >=3.10`, Ruff target `py310`).
- Prefer `from __future__ import annotations` for forward references.
- Avoid language features or typing syntax that requires Python >3.10 unless project metadata is updated.

## OS-agnostic implementation

All code must run correctly on Linux, macOS, and Windows without platform-specific branching unless a feature is genuinely unavailable on a given OS. Use Python stdlib primitives for portability rather than hard-coded strings or platform assumptions.

### Paths

Always use `pathlib.Path` and its methods. Never build paths with string concatenation, `os.path.join`, or `"/"` literals.

```python
# Wrong
path = base + "/" + "subdir"
path = os.path.join(base, "subdir")

# Correct
path = Path(base) / "subdir"
path = Path(base).joinpath("subdir")
```

Resolve and expand paths at the earliest opportunity:

```python
install_path = Path(install_dir).expanduser().resolve()
```

When checking path containment (e.g. to prevent path traversal), use `Path.resolve().relative_to()` rather than string prefix matching — string prefix checks using `"/"` break on Windows:

```python
# Wrong — "/" is not the separator on Windows
if not str(target).startswith(str(base) + "/"):
    raise ValueError(...)

# Correct
try:
    target.resolve().relative_to(base.resolve())
except ValueError:
    raise ValueError("Path escapes expected directory")
```

### Temporary files

`tempfile.NamedTemporaryFile` behaves differently on Windows: the file cannot be reopened or renamed while it is still open. Always use `delete=False`, write inside the `with` block, then close (exit the block) before doing anything with the path:

```python
# Correct — file is closed before Path.replace() is called
with NamedTemporaryFile(delete=False, dir=target_dir, mode="wb") as tmp:
    tmp.write(data)
    tmp_path = Path(tmp.name)

tmp_path.replace(final_path)   # safe: file is closed
```

### Atomic rename

Use `Path.replace()` instead of `os.rename()`. On Windows, `os.rename()` raises `FileExistsError` when the destination already exists; `Path.replace()` overwrites atomically on all platforms:

```python
# Wrong — raises on Windows if destination exists
os.rename(tmp_path, final_path)

# Correct
tmp_path.replace(final_path)
```

Always delete the temp file in a `finally` block in case `replace()` fails:

```python
try:
    tmp_path.replace(final_path)
finally:
    if tmp_path.exists() and tmp_path != final_path:
        tmp_path.unlink(missing_ok=True)
```

### Unix sockets

Unix domain sockets (`*.sock`) do not exist on Windows. Guard any socket-discovery or socket-connect logic explicitly:

```python
if platform.system().lower() == "windows":
    return  # sockets not supported

socket_dir = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
```

`XDG_RUNTIME_DIR` is a Linux-only convention. Fall back to `tempfile.gettempdir()` on other platforms, as the code already does.

### Executable permissions

`os.chmod()` is a no-op on Windows. Restrict `chmod` calls to the binaries and platforms where they are meaningful:

```python
# Only the Unix binary needs the executable bit set;
# the Windows binary (.exe) does not.
if binary_name == "vsce-sign":
    mode = os.stat(tmp_target).st_mode
    os.chmod(tmp_target, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
```

### Platform and architecture detection

`platform.system()` returns `"Linux"`, `"Darwin"` (not `"macOS"`), or `"Windows"` — it is case-sensitive. Always normalise with `.lower()` before comparing:

```python
if platform.system().lower() == "windows":
    ...
```

`platform.machine()` returns different strings for the same underlying hardware across operating systems. Handle all known variants:

| Architecture | Linux / macOS | Windows |
|---|---|---|
| x86-64 | `"x86_64"` | `"AMD64"` |
| ARM 64-bit | `"aarch64"` / `"arm64"` | `"ARM64"` |

Always map all variants to a canonical identifier rather than assuming a single string.

## Documentation and comments

- Add concise, behavior-focused docstrings for non-trivial functions.
- Update docs/comments when behavior changes.
- Avoid noisy comments that restate obvious code.
