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
        **kwargs: Any,
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
            **kwargs: Additional metadata properties to store on the node.

        Example:
            ```python
            gm = GraphManager()
            gm.add_variable("wing_span", value=10.0, lower=5.0, upper=15.0, param_type="range", description="Wing span variable")
            gm.add_variable("material", value="aluminum", choices=["aluminum", "composite"], param_type="choice", value_type="str")
            ```
        """
        import json

        choices_str = json.dumps(choices) if choices is not None else None

        props = {
            "name": name,
            "value": value,
            "lower": lower,
            "upper": upper,
            "param_type": param_type,
            "choices": choices_str,
            "value_type": value_type,
        }

        # Merge extra kwargs
        props.update(kwargs)

        # Remove None values so we don't store them if we don't want to
        props = {k: v for k, v in props.items() if v is not None}

        query = """
        MERGE (v:Variable {name: $name})
        SET v += $props
        """
        self.graph.query(query, params={"name": name, "props": props})

    def add_tool(self, name: str, fidelity: str = "high", **kwargs: Any):
        """Adds a tool node to the graph.

        Args:
            name: The unique name of the engineering tool or solver.
            fidelity: The fidelity level of the tool ("high", "low", etc.). Default is "high".
            **kwargs: Additional metadata properties to store on the node.

        Example:
            ```python
            gm.add_tool("CFD_Solver", fidelity="high", version="1.2.0")
            gm.add_tool("Vortex_Lattice", fidelity="low")
            ```
        """
        props = {"name": name, "fidelity": fidelity}
        props.update(kwargs)

        props = {k: v for k, v in props.items() if v is not None}

        query = """
        MERGE (t:Tool {name: $name})
        SET t += $props
        """
        self.graph.query(query, params={"name": name, "props": props})

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
        query = "MATCH (t:Tool) RETURN t"
        result = self.graph.query(query)
        tools = []
        for r in result.result_set:
            node = r[0]
            tools.append(node.properties)
        return tools

    def get_variables(self) -> list[dict[str, Any]]:
        """Retrieves all variables."""
        import json

        query = "MATCH (v:Variable) RETURN v"
        result = self.graph.query(query)
        vars_list = []
        for r in result.result_set:
            node = r[0]
            props = node.properties

            # Ensure defaults for required fields if they are missing
            if "param_type" not in props:
                props["param_type"] = "continuous"
            if "value_type" not in props:
                props["value_type"] = "float"

            # Handle choices JSON string deserialization
            choices_val = props.get("choices")
            if choices_val and isinstance(choices_val, str) and choices_val != "null":
                try:
                    props["choices"] = json.loads(choices_val)
                except Exception:
                    pass
            elif choices_val == "null":
                props["choices"] = None

            vars_list.append(props)
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
