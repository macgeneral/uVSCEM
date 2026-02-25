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

- This project is sync-first today (`requests`, `argparse`, filesystem operations).
- Do not introduce async/event-loop abstractions unless explicitly required.
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

## Documentation and comments

- Add concise, behavior-focused docstrings for non-trivial functions.
- Update docs/comments when behavior changes.
- Avoid noisy comments that restate obvious code.
