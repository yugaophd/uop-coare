# Releasing

This repository uses [pyproject.toml](/home/yugao/Python/pycoare/pyproject.toml) as the package metadata source of truth.

## Release Checklist

1. Update the version in [pyproject.toml](/home/yugao/Python/pycoare/pyproject.toml).
2. Update any user-facing notes in [README.md](/home/yugao/Python/pycoare/README.md).
3. Run the test suite.
4. Build the source and wheel distributions.
5. Check the built artifacts.
6. Commit the release changes.
7. Create and push a git tag.
8. Upload to PyPI if needed.

## Commands

Run tests from the repository root:

```bash
/home/yugao/Python/pycoare/.venv/bin/python -m pytest -q
```

Build distributions from outside the repository.

Reason: the top-level `build/` directory in this repo shadows the Python `build` package, so `python -m build` fails if run from the repo root.

```bash
cd /tmp
/home/yugao/Python/pycoare/.venv/bin/python -m build /home/yugao/Python/pycoare
```

Check the generated files:

```bash
ls -l /home/yugao/Python/pycoare/dist
```

Optional clean install smoke test:

```bash
python -m venv /tmp/pycoare-release-test
/tmp/pycoare-release-test/bin/pip install /home/yugao/Python/pycoare/dist/*.whl
/tmp/pycoare-release-test/bin/python -c "import uop_coare; print('ok')"
```

Commit and tag:

```bash
git -C /home/yugao/Python/pycoare add pyproject.toml README.md RELEASING.md
git -C /home/yugao/Python/pycoare commit -m "Release vX.Y.Z"
git -C /home/yugao/Python/pycoare tag vX.Y.Z
git -C /home/yugao/Python/pycoare push origin main
git -C /home/yugao/Python/pycoare push origin vX.Y.Z
```

Optional PyPI upload:

```bash
/home/yugao/Python/pycoare/.venv/bin/python -m pip install twine
/home/yugao/Python/pycoare/.venv/bin/python -m twine check /home/yugao/Python/pycoare/dist/*
/home/yugao/Python/pycoare/.venv/bin/python -m twine upload /home/yugao/Python/pycoare/dist/*
```

## Notes

- Release tags should match the package version, for example `v0.1.0`.
- If tests fail, fix that before tagging.
- If `dist/` contains old artifacts, remove them before rebuilding.