from collections.abc import Callable
from typing import Any

from gemseo.mda.factory import MDAFactory

from mdo_framework.core.components import ToolComponent


class GraphProblemBuilder:
    """Builds a GEMSEO MDA/Scenario from a graph schema dictionary."""

    def __init__(self, schema: dict[str, Any]):
        """Initializes the builder with the given graph schema.

        Args:
            schema: A dictionary containing 'tools' and 'variables' definitions.
                   Produced by GraphManager.get_graph_schema().
        """
        self.schema = schema

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
            comp = ToolComponent(name=name, func=func, inputs=inputs, outputs=outputs)
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
                self.default_inputs[var["name"]] = __import__('numpy').atleast_1d(val)

        mda.default_input_data.update(self.default_inputs)
        return mda
