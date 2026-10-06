# Releasing

## Dependency compatibility

The supported optimization stack is currently `ax-platform>=1.2.4,<1.3` with
`botorch>=0.17.2,<0.17.3`. Keep both constraints in `pyproject.toml` and the
resolved versions in `uv.lock` aligned. Ax and BoTorch releases can change
behavior across minor versions, so do not widen either range based only on a
successful dependency resolution.

Before widening the supported range, update both dependency constraints and
regenerate `uv.lock` together. Run the installed-wheel smoke test against the
new pair and run the full test suite. The smoke test installs the built wheel
without consulting the lockfile and performs a real constrained optimization;
it guards the actual distribution metadata and execution path.

## Patch release checklist

- [ ] Confirm the release commit contains only intended changes and the
      `pyproject.toml` constraints and `uv.lock` resolve the same tested pair.
- [ ] Build the distribution with `uv build` and inspect the wheel metadata to
      confirm its Ax and BoTorch requirements.
- [ ] Install the wheel into a clean Python 3.12 environment without `uv.lock`
      and run `tests/smoke/test_installed_wheel.py`.
- [ ] Confirm the smoke test resolves Ax 1.2.4 and BoTorch 0.17.2 and records
      at least two optimization trials.
- [ ] Run `uv sync --frozen --all-extras --dev`, `uv run ruff check .`,
      `uv run pytest tests/`, and `uv run mkdocs build --strict`.
- [ ] Publish the patch release to PyPI using the configured release workflow.
- [ ] As a separate PyPI maintainer action, consider yanking the affected
      `1.0.0` release and communicate the reason in release notes.

Yanking `1.0.0` discourages new unpinned installs from selecting it; it does not
repair environments or installs that explicitly pin `graphmdo==1.0.0`.
