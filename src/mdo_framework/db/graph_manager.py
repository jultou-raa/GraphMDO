from typing import List, Dict, Any
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
        self, name: str, value: Any = None, lower: float = None, upper: float = None
    ):
        """Adds a variable node to the graph."""
        query = f"""
        MERGE (v:Variable {{name: '{name}'}})
        SET v.value = {value if value is not None else 'null'},
            v.lower = {lower if lower is not None else 'null'},
            v.upper = {upper if upper is not None else 'null'}
        """
        self.graph.query(query)

    def add_tool(self, name: str, fidelity: str = "high"):
        """Adds a tool node to the graph."""
        query = f"MERGE (t:Tool {{name: '{name}', fidelity: '{fidelity}'}})"
        self.graph.query(query)

    def connect_tool_to_output(self, tool_name: str, variable_name: str):
        """Connects a tool to an output variable (Tool -> Variable)."""
        query = f"""
        MATCH (t:Tool {{name: '{tool_name}'}}), (v:Variable {{name: '{variable_name}'}})
        MERGE (t)-[:OUTPUTS]->(v)
        """
        self.graph.query(query)

    def connect_input_to_tool(self, variable_name: str, tool_name: str):
        """Connects an input variable to a tool (Variable -> Tool)."""
        query = f"""
        MATCH (v:Variable {{name: '{variable_name}'}}), (t:Tool {{name: '{tool_name}'}})
        MERGE (v)-[:INPUTS_TO]->(t)
        """
        self.graph.query(query)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Retrieves all tools."""
        query = "MATCH (t:Tool) RETURN t.name, t.fidelity"
        result = self.graph.query(query)
        return [{"name": r[0], "fidelity": r[1]} for r in result.result_set]

    def get_variables(self) -> List[Dict[str, Any]]:
        """Retrieves all variables."""
        query = "MATCH (v:Variable) RETURN v.name, v.value, v.lower, v.upper"
        result = self.graph.query(query)
        return [
            {"name": r[0], "value": r[1], "lower": r[2], "upper": r[3]}
            for r in result.result_set
        ]

    def get_tool_inputs(self, tool_name: str) -> List[str]:
        """Retrieves input variables for a specific tool."""
        query = f"""
        MATCH (v:Variable)-[:INPUTS_TO]->(t:Tool {{name: '{tool_name}'}})
        RETURN v.name
        """
        result = self.graph.query(query)
        return [r[0] for r in result.result_set]

    def get_tool_outputs(self, tool_name: str) -> List[str]:
        """Retrieves output variables for a specific tool."""
        query = f"""
        MATCH (t:Tool {{name: '{tool_name}'}})-[:OUTPUTS]->(v:Variable)
        RETURN v.name
        """
        result = self.graph.query(query)
        return [r[0] for r in result.result_set]

    def get_graph_schema(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary representing the entire graph structure.
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
