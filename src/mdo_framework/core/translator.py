"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from gemseo.mda.factory import MDAFactory
from gemseo.utils.discipline import check_disciplines_consistency

from mdo_framework.core.components import ToolComponent
from mdo_framework.core.topology import build_variable_specs
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinition,
    value_to_index,
)
from mdo_framework.schema import (
    FixedParam,
    StateVar,
    StudySchema,
    StudyValidationError,
)
from mdo_framework.validation import validate_registry


def to_design_value(spec: ParameterDefinition | None, value: Any) -> Any:
    """Converts a user-facing value into its GEMSEO representation.

    Choice values become their index; every other value is unchanged.
    Raises ``ParameterValueError`` (a ``ValueError``) for undeclared choices.
    """
    if spec is not None and spec.get("type") == "choice":
        return value_to_index(spec, value)
    return value


def encode_tool_inputs(
    inputs: dict[str, Any],
    specs: dict[str, ParameterDefinition],
) -> dict[str, np.ndarray]:
    """Builds GEMSEO input data from user-facing values."""
    return {
        name: np.atleast_1d(to_design_value(specs.get(name), value))
        for name, value in inputs.items()
    }


class GraphProblemBuilder:
    """Builds a GEMSEO MDA from a typed study schema."""

    def __init__(self, schema: StudySchema):
        """Initializes the builder with the given study schema.

        Args:
            schema: Typed description of the variables and tools, as returned
                by ``GraphManager.get_study_schema()``.
        """
        self.schema = schema
        self.variable_specs = build_variable_specs(schema)

    def build_problem(self, tool_registry: Mapping[str, Callable]) -> Any:
        """Constructs a GEMSEO MDA from the study schema.

        Fixed parameter values and coupling initial guesses become the default
        inputs of the MDA; design variables have no default.

        Args:
            tool_registry: Dictionary mapping tool names to Python functions.

        Returns:
            An instantiated GEMSEO MDA Discipline object.

        Raises:
            StudyValidationError: If a tool has no registered function or the
                registered function cannot take the tool's inputs by keyword.
                Every bad tool is reported in the same error.
            ValueError: If two tools produce the same output.
        """
        report = validate_registry(self.schema, tool_registry)
        if not report.valid:
            raise StudyValidationError(report)

        disciplines = [
            ToolComponent(
                name=tool.name,
                func=tool_registry[tool.name],
                inputs=tool.inputs,
                outputs=tool.outputs,
                specs={
                    in_name: self.variable_specs[in_name]
                    for in_name in tool.inputs
                    if in_name in self.variable_specs
                },
            )
            for tool in self.schema.tools
        ]
        check_disciplines_consistency(disciplines, False, True)

        # MDAChain picks the sub-MDAs itself (MDAJacobi for coupled disciplines).
        mda = MDAFactory().create("MDAChain", disciplines=disciplines)

        for name, value in self._seed_values().items():
            if name in mda.input_grammar:
                mda.default_input_data[name] = value

        return mda

    def _seed_values(self) -> dict[str, np.ndarray]:
        defaults = {}
        for variable in self.schema.variables:
            if isinstance(variable, FixedParam):
                defaults[variable.name] = np.atleast_1d(variable.value)
            elif isinstance(variable, StateVar) and variable.initial_guess is not None:
                defaults[variable.name] = np.atleast_1d(
                    np.asarray(variable.initial_guess, dtype=float)
                )
        return defaults
