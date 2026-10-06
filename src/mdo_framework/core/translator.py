"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from gemseo.mda.factory import MDAFactory

from mdo_framework.core.components import ToolComponent
from mdo_framework.core.topology import build_variable_specs
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinition,
    value_to_index,
)


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
    """Builds a GEMSEO MDA/Scenario from a graph schema dictionary."""

    def __init__(self, schema: dict[str, Any]):
        """Initializes the builder with the given graph schema.

        Args:
            schema: A dictionary containing 'tools' and 'variables' definitions.
                   Produced by GraphManager.get_graph_schema().
        """
        self.schema = schema
        self.variable_specs = build_variable_specs(schema)

    def build_problem(self, tool_registry: dict[str, Callable]) -> Any:
        """Constructs a GEMSEO MDA from the parsed schema.

        Args:
            tool_registry: Dictionary mapping tool names to Python functions.

        Returns:
            An instantiated GEMSEO MDA Discipline object.
        """
        tools = self.schema.get("tools", [])
        disciplines = []

        # Add components
        for tool in tools:
            name = tool["name"]
            func = tool_registry.get(name)

            if not func:
                raise ValueError(f"Tool function for '{name}' not found in registry.")

            inputs = tool.get("inputs", [])
            outputs = tool.get("outputs", [])

            # Wrap the function in our custom GEMSEO Discipline
            comp = ToolComponent(
                name=name,
                func=func,
                inputs=inputs,
                outputs=outputs,
                specs={
                    in_name: self.variable_specs[in_name]
                    for in_name in inputs
                    if in_name in self.variable_specs
                },
            )
            disciplines.append(comp)

        # Create an MDA (Multidisciplinary Design Analysis) to handle the coupling
        # We use 'MDAChain' by default which can handle sequential execution
        # and incorporates an 'MDAGaussSeidel' if cycles exist.
        mda_factory = MDAFactory()
        mda = mda_factory.create("MDAChain", disciplines=disciplines)

        # We can extract default values from schema and store them
        # to be used later in execution
        self.default_inputs = {}
        variables = self.schema.get("variables", [])
        for var in variables:
            val = var.get("value")
            if val is not None:
                spec = self.variable_specs[var["name"]]
                self.default_inputs[var["name"]] = np.atleast_1d(
                    to_design_value(spec, val)
                )

        for var_name, var_val in self.default_inputs.items():
            if var_name in mda.input_grammar:
                mda.default_input_data[var_name] = var_val

        return mda
