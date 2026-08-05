# Conda packaging

Two recipes for the same package.

| file | used by | for |
|---|---|---|
| `meta.yaml` | `conda-build` | the bioconda submission form |
| `recipe.yaml` | `rattler-build` | building locally, today |

Both are unpublished. Authorship needs settling with collaborators before
anything is submitted.

## Build and install locally

```sh
rattler-build build -r conda-recipe/recipe.yaml -m variants.yaml \
    -c conda-forge -c bioconda --output-dir /tmp/cbuild

conda create -n fastaai_test -c /tmp/cbuild -c conda-forge -c bioconda fastaai
```

`variants.yaml` just pins the interpreter:

```yaml
python:
  - "3.11"
```

Without it the solve tries every Python ABI at once and fails. The extension is
abi3, so the pin only selects which interpreter to build against.

Verified end to end from the installed environment: gzipped genome FASTAs in,
AAI out, no source tree and no `--hmm` — 3 Firmicutes preprocessed in 5 s, the
bundled 122 models found at
`$CONDA_PREFIX/lib/python3.11/site-packages/fastaai/data/`.

## Why two recipes

`conda-build` 26.1.0 cannot solve its own build environment here. It reports the
*virtual* packages as dependencies needing to be built:

```
DependencyNeedsBuildingError: Unsatisfiable dependencies for platform linux-64:
  {'__unix', '__linux', '__archspec', '__glibc', '__cuda', '__conda'}
```

Confirmed as a conda-build fault rather than a recipe one: `conda render`
succeeds, `conda create` solves and installs normally, and setting
`CONDA_OVERRIDE_CUDA=""` removes `__cuda` from that list — so the solver is not
seeing the virtual packages conda injects everywhere else. `rattler-build`
builds the same recipe without complaint.

`meta.yaml` is kept current anyway, because bioconda takes that form.

## Before submitting

- **Authorship.** `pyproject.toml` lists Kenji Gerhardt alone; v1 credits four
  people. Settle this first.
- Replace `source: path:` with a `url` + `sha256` of a tagged release.
- Restore `{{ compiler('rust') }}` in `meta.yaml`; it is spelled out here only
  because a local build lacks conda-forge's pinning config.
