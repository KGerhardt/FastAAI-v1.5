"""Shared test fixtures.

The CLI writes its output root — `FastAAI/` — into the working directory, by
design: an HPC job is given a directory and is expected to stay in it. That
makes the working directory part of what every CLI test exercises, so each test
gets its own and none of them can leave anything in the repository.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in its own directory.

    Autouse rather than opt-in: a test that forgets it would silently write
    `FastAAI/` into the source tree, and the failure shows up as pollution
    somewhere else rather than as a failing test.
    """
    workdir = tmp_path / "cwd"
    workdir.mkdir(exist_ok=True)
    monkeypatch.chdir(workdir)
    yield workdir
