"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Any

import numpy as np
from gemseo.core.discipline import Discipline

from mdo_framework.core.translator import encode_tool_inputs
from mdo_framework.optimization.parameter_codec import ParameterDefinition


class LocalEvaluator:
    """Evaluates the design parameters locally using a GEMSEO MDA instance.

    Args:
        problem: An instantiated GEMSEO MDA (or Discipline) object.
        variable_specs: Optional parameter definitions per variable name
            (``GraphProblemBuilder.variable_specs``). Required to pass declared
            choice values to ``evaluate``; they are converted to GEMSEO indices.
    """

    def __init__(
        self,
        problem: Discipline,
        variable_specs: dict[str, ParameterDefinition] | None = None,
    ):
        self.problem = problem
        self.variable_specs = variable_specs or {}

    def evaluate(
        self,
        parameters: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, float]:
        # GEMSEO uses a dictionary with string keys and numpy array values for local_data
        input_data = encode_tool_inputs(parameters, self.variable_specs)

        # We need to provide all required inputs for the MDA, not just parameters.
        # This will be passed and merged internally by execute.
        output_data = self.problem.execute(input_data)

        results = {}
        for obj in objectives:
            val = output_data.get(obj)
            if val is None:
                raise KeyError(f"Missing objective or constraint output: {obj}")
            results[obj] = (
                float(val[0])
                if isinstance(val, np.ndarray) and val.size > 0
                else float(val)
            )
        return results
