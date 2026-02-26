import shutil
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def remove_created_directories_on_pass(request):
    """
    Remove directories created during the test if the test passes.
    This is useful to clean up OpenMDAO output directories or other
    directories created during tests, while keeping them for debugging
    if the test fails.
    """
    cwd = Path.cwd()
    # Snapshot of directories before the test runs
    dirs_before = {d.name for d in cwd.iterdir() if d.is_dir()}

    yield

    rep_call = getattr(request.node, "rep_call", None)

    # If test passed, we clean up
    if rep_call is not None and rep_call.passed:
        dirs_after = {d.name for d in cwd.iterdir() if d.is_dir()}
        new_dirs = dirs_after - dirs_before

        safe_to_keep = {".pytest_cache", ".ruff_cache", "__pycache__", ".venv", ".git"}

        for d_name in new_dirs:
            if d_name not in safe_to_keep:
                dir_to_remove = cwd / d_name
                try:
                    shutil.rmtree(dir_to_remove, ignore_errors=True)
                except BaseException:
                    pass
