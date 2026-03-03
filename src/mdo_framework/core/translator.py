from typing import Any, Callable

import openmdao.api as om

from mdo_framework.core.components import ToolComponent


class GraphProblemBuilder:
    """
    Builds an OpenMDAO Problem from a graph schema dictionary.
    """

    def __init__(self, schema: dict[str, Any]):
        """
        Initializes the builder with the given graph schema.

        Args:
            schema: Dictionary representing tools, inputs, and outputs.
        """
        """
        Args:
            schema: A dictionary containing 'tools' and 'variables' definitions.
                   Produced by GraphManager.get_graph_schema().
        """
        self.schema = schema

    def build_problem(self, tool_registry: dict[str, Callable]) -> om.Problem:
        """
        Constructs an OpenMDAO problem from the parsed schema.

        Args:
            tool_registry: Mapping of tool names to Python callables.

        Returns:
            An instantiated OpenMDAO Problem object.
        """
        """
        Constructs and sets up the OpenMDAO problem.

        Args:
            tool_registry: Dictionary mapping tool names to Python functions.
        """
        prob = om.Problem()
        model = prob.model

        tools = self.schema.get("tools", [])

        # Add components
        for tool in tools:
            name = tool["name"]
            func = tool_registry.get(name)

            if not func:
                # If function not found, perhaps it's a surrogate or dummy?
                # For this implementation, we require it in registry.
                raise ValueError(f"Tool function for '{name}' not found in registry.")

            inputs = tool.get("inputs", [])
            outputs = tool.get("outputs", [])

            comp = ToolComponent(name=name, func=func, inputs=inputs, outputs=outputs)

            # Promote all variables to connect by name automatically
            # This matches the graph topology where a Variable node is a shared entity.
            model.add_subsystem(name, comp, promotes=["*"])

        prob.setup()

        # Initialize variables if values are present in schema
        variables = self.schema.get("variables", [])
        for var in variables:
            val = var.get("value")
            if val is not None:
                try:
                    # Only set if it's an input/independent variable or initial guess
                    prob.set_val(var["name"], val)
                except Exception:
                    # Variable might not be promoted or connected, ignore
                    pass

        return prob
