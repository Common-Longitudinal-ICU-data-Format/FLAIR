# Publishing `flair-benchmark` to PyPI

> Maintainer guide. This file is **not** shipped in the package (the sdist/wheel `include` list is explicit — see `pyproject.toml [tool.hatch.build.targets.sdist]`).

- **Dist name:** `flair-benchmark` (the bare name `flair` is taken by flairNLP)
- **Import name:** `flair_benchmark`
- **Current version:** `0.0.1` (first release)
- **Build backend:** Hatchling, version read dynamically from `flair_benchmark/_constants.py::__version__`

------------------------------------------------------------------------

## What ships to the library (and what does NOT)

The package is **source only** — no docs, no data, no notebooks. Controlled by `pyproject.toml`:

``` toml
[tool.hatch.build.targets.wheel]
packages = ["flair_benchmark"]

[tool.hatch.build.targets.sdist]
include = ["/flair_benchmark", "/README.md", "/LICENSE", "/pyproject.toml"]
```

### Ships (in wheel + sdist)

| Path | Why |
|---------------------------------------|---------------------------------|
| `flair_benchmark/**/*.py` | the library source (cli, tasks, report, features, meds_etl) |
| `flair_benchmark/meds_etl/concepts/*.yaml` | runtime data — concept maps the ETL loads at runtime |
| `README.md` | PyPI long description |
| `LICENSE` | Apache-2.0 |
| `pyproject.toml` | build metadata (sdist only) |

### Excluded on purpose (stays in the repo, never published)

| Path | Reason |
|-------------------------------|-----------------------------------------|
| `docs/` (101, paper PDFs, task_schematics, lab_show) | docs not wanted in package; \~17 MB bloat |
| `assets/` (FlairWorkflow.gif) | repo-only marketing asset |
| `uv.lock` | lockfile is for dev, not consumers |
| `clif_config_mimic.json`, `feature_config.yaml` | local/site config, not part of the lib |
| `tests/` | not needed by consumers |
| `flair_baseline/` | separate package (being split to its own repo) |

> `init-config` scaffolds fresh `clif_config.json` + `feature_config.yaml` templates for users at runtime, so the repo's own config files don't need to ship.

------------------------------------------------------------------------

## Release workflow

### 0. Prereqs

- A PyPI account + API token (and a TestPyPI token for the dry-run).
- `uv` installed.
- `flair_benchmark/` source present in the working tree (it was restored from pre-split commit `e12d527`; the origin tip `347dbd1` had removed it).

### 1. Bump the version (single source of truth)

Edit **only** `flair_benchmark/_constants.py`:

``` python
__version__ = "0.0.1"
```

`pyproject.toml` reads it dynamically; `__init__.py` re-exports it; every report JSON stamps it as `flair_version` (`report/__init__.py`, `_table1.py`). No other file needs editing.

Sanity check — no stray hardcoded version (ignore the `ruff>=0.1.0` dev dep):

``` bash
grep -rn "__version__" flair_benchmark/_constants.py
```

### 2. Sanitize

- No secrets / keys / PHI in the package source (verified).
- `.gitignore` excludes `*.csv`, `*.parquet`, `data/`, `*.pkl`, `*.h5`.
- Site config files carry local paths but are **not** in the package `include` list, so they never ship.

### 3. Build

``` bash
cd /Users/sudo_sage/Documents/WORK/FLAIR_PROJECT/FLAIR
rm -rf dist build
uv build
```

Produces `dist/flair_benchmark-<version>.tar.gz` and `...-py3-none-any.whl`.

### 4. Verify the artifacts

``` bash
# no docs/assets/lockfile leaked into the sdist
tar -tzf dist/*.tar.gz | grep -iE 'docs/|assets/|uv.lock' && echo BAD || echo OK

# metadata + README render check
uv run --with twine twine check dist/*

# isolated install: version + task discovery + CLI entry point
uv run --no-project --with dist/*.whl python -c \
  "import flair_benchmark as f; print(f.__version__); print(f.list_tasks())"
uv run --no-project --with dist/*.whl flair --help

# full test suite (synthetic data, no network)
uv run --extra dev pytest -q
```

Expected: `twine check` PASSED, version prints `0.0.1`, 5 tasks listed, tests green.

### 5. Publish — TestPyPI dry-run first

``` bash
uv publish --publish-url https://test.pypi.org/legacy/ --token <TESTPYPI_TOKEN>

# confirm it installs from TestPyPI
uv run --no-project \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --with flair-benchmark flair --help
```

### 6. Publish — real PyPI

``` bash
uv publish --token <PYPI_TOKEN>
```

### 7. Post-release

- Tag the release: `git tag v0.0.1 && git push origin v0.0.1`
- A version can never be re-uploaded to PyPI — to fix a bad release, bump to the next version and republish.

------------------------------------------------------------------------

## Notes / gotchas

- **Version is a downgrade vs. old source (`0.1.0` → `0.0.1`).** Fine here because nothing is on PyPI yet; this is the first upload.
- The library lives in this repo alongside `flair_baseline/` (own repo) and the website (own repo) per the `split` work — only `flair_benchmark/` is published.
- Optional, not blocking: add `flair_benchmark/py.typed` if shipping type hints; add a `CHANGELOG.md`.