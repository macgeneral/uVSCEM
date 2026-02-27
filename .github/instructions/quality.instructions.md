# Quality Gates Instruction File

Apply this checklist before marking work complete.

## Type and style requirements

- Keep code compatible with Python 3.10+.
- Use modern type syntax (`str | None`, `list[str]`, `dict[str, Any]`) where practical.
- Keep docstrings concise and behavior-focused.
- Keep implementation modules and test modules separate.
- Keep changes minimal and aligned to existing project style.

## Testing requirements

- Prefer deterministic tests that validate behavior/spec.
- Cover extension resolution, API response handling, and CLI-visible outcomes.
- Mock external boundaries (Marketplace HTTP, filesystem-heavy operations, subprocess calls) in unit tests.
- Add/update tests when behavior changes; avoid test-only compatibility shims.

## Validation commands (repo root)

Activate environment first:

`source .venv/bin/activate`

1. `uv sync --group dev`
2. `uv run ruff check .`
3. `uv run ty check`
4. `uv run pytest src`
5. `uv run pytest src --slow --no-cov`

## Completion criteria

- No lint/type/test regressions from your change.
- No new warnings/errors introduced by your changes.
- No secrets/credentials in logs or error output.
- Documentation updated when command behavior or configuration expectations change.

## Refactor policy

Hard rules:

- Reuse existing helpers before adding new logic.
- Avoid copy/paste duplication.
- Keep internal APIs simple; avoid unnecessary compatibility wrappers.

Guidelines:

- Prefer composition over inheritance.
- Prefer small single-purpose functions over deeply nested control flow.
- Keep file sizes reasonable and split by responsibility when needed.
