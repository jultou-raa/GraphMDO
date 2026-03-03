from typing import Any

from mdo_framework.db.client import FalkorDBClient


class GraphManager:
    def __init__(self):
        self.client = FalkorDBClient()
        self.graph = self.client.get_graph()

    def clear_graph(self):
        """Clears the entire graph."""
        query = "MATCH (n) DETACH DELETE n"
        self.graph.query(query)

    def add_variable(
        self,
        name: str,
        value: Any = None,
        lower: float = None,
        upper: float = None,
        param_type: str = "continuous",
        choices: list = None,
        value_type: str = "float",
    ):
        """Adds a variable node to the graph.

        Args:
            name: The unique name of the variable.
            value: The initial or default value of the variable.
            lower: The lower bound for optimization (applicable if param_type is range/continuous).
            upper: The upper bound for optimization (applicable if param_type is range/continuous).
            param_type: The type of parameter ("continuous", "range", "choice", or "fixed").
            choices: A list of valid options if the param_type is "choice".
            value_type: The underlying data type ("float", "int", "str").

        Example:
            ```python
            gm = GraphManager()
            gm.add_variable("wing_span", value=10.0, lower=5.0, upper=15.0, param_type="range")
            gm.add_variable("material", value="aluminum", choices=["aluminum", "composite"], param_type="choice", value_type="str")
            ```
        """
        import json

        choices_str = json.dumps(choices) if choices is not None else "null"
        query = f"""
        MERGE (v:Variable {{name: '{name}'}})
        SET v.value = {value if value is not None else "null"},
            v.lower = {lower if lower is not None else "null"},
            v.upper = {upper if upper is not None else "null"},
            v.param_type = '{param_type}',
            v.choices = '{choices_str}',
            v.value_type = '{value_type}'
        """
        self.graph.query(query)

    def add_tool(self, name: str, fidelity: str = "high"):
        """Adds a tool node to the graph.

        Args:
            name: The unique name of the engineering tool or solver.
            fidelity: The fidelity level of the tool ("high", "low", etc.). Default is "high".

        Example:
            ```python
            gm.add_tool("CFD_Solver", fidelity="high")
            gm.add_tool("Vortex_Lattice", fidelity="low")
            ```
        """
        query = f"MERGE (t:Tool {{name: '{name}', fidelity: '{fidelity}'}})"
        self.graph.query(query)

    def connect_tool_to_output(self, tool_name: str, variable_name: str):
        """Connects a tool to an output variable (Tool -> Variable).

        Args:
            tool_name: The name of the source tool.
            variable_name: The name of the target output variable.

        Example:
            ```python
            gm.connect_tool_to_output("CFD_Solver", "drag_coefficient")
            ```
        """
        query = f"""
        MATCH (t:Tool {{name: '{tool_name}'}}), (v:Variable {{name: '{variable_name}'}})
        MERGE (t)-[:OUTPUTS]->(v)
        """
        self.graph.query(query)

    def connect_input_to_tool(self, variable_name: str, tool_name: str):
        """Connects an input variable to a tool (Variable -> Tool).

        Args:
            variable_name: The name of the source input variable.
            tool_name: The name of the target tool consuming the variable.

        Example:
            ```python
            gm.connect_input_to_tool("wing_span", "CFD_Solver")
            ```
        """
        query = f"""
        MATCH (v:Variable {{name: '{variable_name}'}}), (t:Tool {{name: '{tool_name}'}})
        MERGE (v)-[:INPUTS_TO]->(t)
        """
        self.graph.query(query)

    def get_tools(self) -> list[dict[str, Any]]:
        """Retrieves all tools."""
        query = "MATCH (t:Tool) RETURN t.name, t.fidelity"
        result = self.graph.query(query)
        return [{"name": r[0], "fidelity": r[1]} for r in result.result_set]

    def get_variables(self) -> list[dict[str, Any]]:
        """Retrieves all variables."""
        import json

        query = "MATCH (v:Variable) RETURN v.name, v.value, v.lower, v.upper, v.param_type, v.choices, v.value_type"
        result = self.graph.query(query)
        vars_list = []
        for r in result.result_set:
            choices_val = r[5]
            if choices_val and choices_val != "null" and isinstance(choices_val, str):
                try:
                    choices_val = json.loads(choices_val)
                except Exception:
                    pass
            elif choices_val == "null":
                choices_val = None

            vars_list.append(
                {
                    "name": r[0],
                    "value": r[1],
                    "lower": r[2],
                    "upper": r[3],
                    "param_type": r[4] if r[4] else "continuous",
                    "choices": choices_val,
                    "value_type": r[6] if len(r) > 6 and r[6] else "float",
                }
            )
        return vars_list

    def get_tool_inputs(self, tool_name: str) -> list[str]:
        """Retrieves input variables for a specific tool."""
        query = f"""
        MATCH (v:Variable)-[:INPUTS_TO]->(t:Tool {{name: '{tool_name}'}})
        RETURN v.name
        """
        result = self.graph.query(query)
        return [r[0] for r in result.result_set]

    def get_tool_outputs(self, tool_name: str) -> list[str]:
        """Retrieves output variables for a specific tool."""
        query = f"""
        MATCH (t:Tool {{name: '{tool_name}'}})-[:OUTPUTS]->(v:Variable)
        RETURN v.name
        """
        result = self.graph.query(query)
        return [r[0] for r in result.result_set]

    def get_graph_schema(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary representing the entire graph structure.

        Returns:
            A dictionary containing 'tools' and 'variables' lists defining the topology.

        Example:
            ```python
            schema = gm.get_graph_schema()
            print(schema["tools"][0]["name"])
            ```
        """
        tools = self.get_tools()
        variables = self.get_variables()
        schema = {"tools": [], "variables": variables}

        for tool in tools:
            name = tool["name"]
            inputs = self.get_tool_inputs(name)
            outputs = self.get_tool_outputs(name)
            schema["tools"].append(
                {
                    "name": name,
                    "fidelity": tool["fidelity"],
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )

        return schema
