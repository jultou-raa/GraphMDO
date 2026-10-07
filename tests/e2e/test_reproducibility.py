"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Seeded runs are reproducible and side-effect free.
"""

import pytest

pytestmark = pytest.mark.e2e


def test_same_seed_same_run(shifted_paraboloid):
    first, _ = shifted_paraboloid()
    second, _ = shifted_paraboloid()

    first_history = first.optimize(n_steps=3, n_init=3)["history"]
    second_history = second.optimize(n_steps=3, n_init=3)["history"]

    assert first_history == second_history


@pytest.mark.xfail(
    strict=True, reason="#54: optimize() writes XDSM and plots to the cwd"
)
def test_optimize_writes_no_files(shifted_paraboloid, tmp_path):
    optimizer, _ = shifted_paraboloid()

    optimizer.optimize(n_steps=1, n_init=1)

    assert list(tmp_path.iterdir()) == []
