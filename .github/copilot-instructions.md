# uVSCEM Copilot Instructions

This repository is Python 3.10+. Optimize for small, composable modules with explicit boundaries.

## Core engineering principles

- Favor KISS-first design.
	- Prefer straightforward, easily understood solutions over layered abstractions.
	- Optimize for readability and maintainability.
- Prefer composition over inheritance.
- Keep APIs lean.
	- Add parameters and return values only when they are needed.
- Avoid duplication.
	- Reuse helpers before introducing new code paths.
- Keep changes surgical.
	- Do not refactor unrelated areas.

## Project structure conventions

- Package root: `src/uvscem/*`.
- Marketplace API logic: `src/uvscem/api_client.py`.
- VSCode socket/CLI discovery logic: `src/uvscem/code_manager.py`.
- Extension orchestration + CLI command wiring: `src/uvscem/extension_manager.py`.
- Tests: `src/tests/*`.

## Runtime model and behavior

- Use `asyncio` for all IO-bound operations (HTTP, subprocess, filesystem); wrap sync helpers with `asyncio.to_thread` where needed.
- Do not use async for CPU-bound or purely in-memory logic.
- Preserve proxy-aware download behavior.
- Preserve dependency + extension-pack resolution semantics.
- Keep compatibility with `devcontainer.json` extension lists (`customizations.vscode.extensions`).
- Preserve VSCode server path conventions under `~/.vscode-server` unless behavior changes are requested.
 - Use OS-agnostic paths, sockets, and temporary files by leveraging Python stdlib primitives (for example `pathlib`, `tempfile`, and `socket` internals) rather than hard-coded path strings or platform-specific separators.

## Typing and style

- Target Python 3.10+.
- Prefer modern type syntax (`str | None`, `list[str]`, `dict[str, Any]`).
- Keep type hints on public functions and non-trivial internals.
- Keep docstrings concise and behavior-focused.
- Avoid noisy comments that restate obvious code.

## Error handling and security

- Never leak credentials or tokens in logs/exceptions.
- Fail with clear, actionable errors.
- Keep retries and fallback behavior explicit and easy to reason about.

## Testing and validation

- Keep tests deterministic and behavior-oriented.
- Mock external boundaries (HTTP, subprocess, filesystem-heavy paths) in unit tests.
- When behavior changes, update or add tests in `src/tests/*`.

Run from repo root:

1. `uv sync --group dev`
2. `uv run ruff check .`
3. `uv run ruff format .`
4. `uv run ty check`
5. `uv run pytest src`

Quality gate policy:

- Always run the full quality gate after changes (unless the user explicitly asks not to):
	1. `uv run ruff check .`
	2. `uv run ruff format .`
	3. `uv run ty check`
	4. `uv run pytest src`

## Change strategy

- Make minimal diffs.
- Preserve existing naming/style patterns unless change is necessary.
- Keep documentation aligned with behavior/configuration changes.

## Exceptions

- If the user explicitly requests a different approach, follow user intent.
