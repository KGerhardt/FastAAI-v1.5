# Conda packaging

Two recipes for the same package.

| file | used by | for |
|---|---|---|
| `meta.yaml` | `conda-build` | the bioconda submission form |
| `recipe.yaml` | `rattler-build` | building locally, today |

`meta.yaml` is the form submitted to bioconda for 1.5.0. It builds from the
release sdist, so it needs no working tree; `recipe.yaml` still builds from
`../fastaai1_5` and is the one to use while developing.

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

## The 1.5.0 submission

`meta.yaml` points at the **release sdist**, not the auto-generated tag
tarball:

```
https://github.com/KGerhardt/FastAAI-v1.5/releases/download/1.5.0/fastaai-1.5.0.tar.gz
```

That distinction matters. The package lives in `fastaai1_5/`, one level down
from the repo root, so GitHub's own tag tarball would put it a directory below
where conda-build looks and the build script would have to descend into it. The
sdist that `maturin sdist` produces from inside `fastaai1_5/` has the package at
its root, which is why the recipe needs no `cd`. Build the asset with

```sh
cd fastaai1_5 && maturin sdist --out /some/dir
```

and attach that exact file to the release — `maturin sdist` is not
byte-reproducible across runs, so the `sha256` in the recipe is the hash of the
uploaded artifact, not of a rebuild.

To verify a submission before opening the pull request, copy `recipe.yaml`,
swap its `source: path:` for the same `url` + `sha256`, and build that. Done for
1.5.0: it builds and every test command passes.
