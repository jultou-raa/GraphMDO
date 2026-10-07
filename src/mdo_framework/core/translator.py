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
from mdo_framework.core.dependencies import walk_dependencies
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
    ToolNode,
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

        Each tool gets its own defaults: the value of its fixed parameters, the
        initial guess of its state variables, and 0.0 for a coupling variable
        without an initial guess. Design variables have no default. The MDA
        gathers the defaults of its disciplines.

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

        couplings = set(walk_dependencies(self.schema, []).couplings)
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
                defaults=self._tool_defaults(tool, couplings),
                deterministic=tool.deterministic,
                arg_map=tool.arg_map,
            )
            for tool in self.schema.tools
        ]
        check_disciplines_consistency(disciplines, False, True)

        # MDAChain picks the sub-MDAs itself (MDAJacobi for coupled disciplines).
        return MDAFactory().create("MDAChain", disciplines=disciplines)

    def _tool_defaults(self, tool: ToolNode, couplings: set[str]) -> dict[str, Any]:
        variables = {variable.name: variable for variable in self.schema.variables}
        defaults: dict[str, Any] = {}
        for name in tool.inputs:
            variable = variables[name]
            if isinstance(variable, FixedParam):
                defaults[name] = to_design_value(
                    self.variable_specs.get(name), variable.value
                )
            elif isinstance(variable, StateVar):
                if variable.initial_guess is not None:
                    defaults[name] = np.asarray(variable.initial_guess, dtype=float)
                elif name in couplings:
                    defaults[name] = 0.0
        return defaults
