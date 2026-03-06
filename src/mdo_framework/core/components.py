"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Callable

from gemseo.core.discipline import Discipline
import numpy as np


class ToolComponent(Discipline):
    """Generic GEMSEO discipline that wraps a Python function."""

    def __init__(
        self,
        name: str,
        func: Callable,
        inputs: list[str],
        outputs: list[str],
        derivatives: bool = False,
    ):
        """Initializes the generic GEMSEO tool component.

        Args:
            name: The name of the discipline.
            func: The Python callable executing the tool logic.
            inputs: List of input variable names.
            outputs: List of output variable names.
            derivatives: Whether the function provides analytical derivatives (default False).
        """
        super().__init__(name=name)
        self.func = func
        self._inputs_list = inputs
        self._outputs_list = outputs
        self._derivatives = derivatives

        # GEMSEO Grammars require us to define input/output names
        self.input_grammar.update_from_names(self._inputs_list)
        self.output_grammar.update_from_names(self._outputs_list)

        # GEMSEO expects default values to be set in default_inputs if they exist
        self.default_inputs = {
            in_name: np.array([0.0]) for in_name in self._inputs_list
        }

    def _run(self, **kwargs) -> None:
        """Executes the wrapped function using data from self.local_data and stores results.

        Expects the wrapped function to return a dictionary mapping output names
        to their computed values, or a single value for single outputs, or a tuple.
        """
        # Prepare inputs as a dictionary (GEMSEO stores arrays in local_data)
        input_vals = {name: self.local_data[name] for name in self._inputs_list}

        # Execute the function
        try:
            result = self.func(**input_vals)
        except TypeError:
            # Fallback if function expects positional arguments (simple wrappers)
            # Unpack the arrays if they are single elements and the function expects scalars
            positional_args = [
                val[0] if isinstance(val, np.ndarray) and val.size == 1 else val
                for val in input_vals.values()
            ]
            result = self.func(*positional_args)

        # Map results to outputs inside self.local_data
        if len(self._outputs_list) == 1:
            output_name = self._outputs_list[0]
            self.local_data[output_name] = np.atleast_1d(result)
        elif isinstance(result, dict):
            for name in self._outputs_list:
                self.local_data[name] = np.atleast_1d(result[name])
        else:
            # If result is a tuple/list, assume order matches outputs
            for i, name in enumerate(self._outputs_list):
                self.local_data[name] = np.atleast_1d(result[i])

    def _compute_jacobian(
        self, inputs: list[str] = None, outputs: list[str] = None
    ) -> None:
        """Computes the analytical derivatives if provided."""
        if self._derivatives:
            # Placeholder for exact jacobian
            pass
        else:
            # GEMSEO handles finite differences automatically if we call self.set_jacobian_approximation()
            # which is typically done outside or at initialization.
            pass
