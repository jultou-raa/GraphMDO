"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from mdo_framework.optimization import errors, optimizer
from mdo_framework.optimization.errors import (
    OptimizationConfigurationError,
    OptimizationExecutionError,
)


def test_optimizer_reexports_the_same_classes():
    assert optimizer.OptimizationConfigurationError is OptimizationConfigurationError
    assert optimizer.OptimizationExecutionError is OptimizationExecutionError
    assert issubclass(errors.OptimizationConfigurationError, ValueError)
    assert issubclass(errors.OptimizationExecutionError, RuntimeError)


def test_execution_error_keeps_the_partial_result():
    partial = {"history": [{"index": 0, "status": "completed"}]}

    error = OptimizationExecutionError("execution service lost", partial_result=partial)

    assert str(error) == "execution service lost"
    assert error.partial_result == partial


def test_execution_error_has_no_partial_result_by_default():
    assert OptimizationExecutionError("boom").partial_result is None
