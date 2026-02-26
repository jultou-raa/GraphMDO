import openmdao.api as om
from typing import Dict, Callable, Any
from mdo_framework.core.components import ToolComponent


class GraphProblemBuilder:
    """
    Builds an OpenMDAO Problem from a graph schema dictionary.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Args:
            schema: A dictionary containing 'tools' and 'variables' definitions.
                   Produced by GraphManager.get_graph_schema().
        """
        self.schema = schema

    def build_problem(self, tool_registry: Dict[str, Callable]) -> om.Problem:
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
