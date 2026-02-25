import openmdao.api as om
from typing import Dict, Callable
from mdo_framework.db.graph_manager import GraphManager
from mdo_framework.core.components import ToolComponent


class GraphProblemBuilder:
    """
    Builds an OpenMDAO Problem from the FalkorDB graph.
    """

    def __init__(self, graph_manager: GraphManager):
        self.gm = graph_manager

    def build_problem(self, tool_registry: Dict[str, Callable]) -> om.Problem:
        """
        Constructs and sets up the OpenMDAO problem.

        Args:
            tool_registry: Dictionary mapping tool names to Python functions.
        """
        prob = om.Problem()
        model = prob.model

        tools = self.gm.get_tools()

        # Add components
        for tool in tools:
            name = tool["name"]
            func = tool_registry.get(name)

            if not func:
                # If function not found, perhaps it's a surrogate or dummy?
                # For this implementation, we require it in registry.
                raise ValueError(f"Tool function for '{name}' not found in registry.")

            inputs = self.gm.get_tool_inputs(name)
            outputs = self.gm.get_tool_outputs(name)

            comp = ToolComponent(name=name, func=func, inputs=inputs, outputs=outputs)

            # Promote all variables to connect by name automatically
            # This matches the graph topology where a Variable node is a shared entity.
            model.add_subsystem(name, comp, promotes=["*"])

        prob.setup()
        return prob
