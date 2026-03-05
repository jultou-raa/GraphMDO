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

    def add_node(self, kind: str, name: str, **kwargs: Any):
        """Adds a generic node to the graph.

        Args:
            kind: The type of node (e.g., 'Tool' or 'Variable').
            name: The unique name of the node.
            **kwargs: Additional metadata properties to store on the node.
        """
        props = {"name": name}
        props.update(kwargs)

        # Remove None values so we don't store them if we don't want to
        props = {k: v for k, v in props.items() if v is not None}

        # Dynamically build the query with the specific label.
        # Ensure 'kind' only contains letters to prevent Cypher injection issues.
        if not kind.isalpha():
            raise ValueError(f"Invalid node kind: {kind}")

        query = (
            "\n"
            "        MERGE (n:" + kind + " {name: $name})\n"
            "        SET n += $props\n"
            "        "
        )
        self.graph.query(query, params={"name": name, "props": props})

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
        self.add_node(
            kind="Variable",
            name=name,
            value=value,
            lower=lower,
            upper=upper,
            param_type=param_type,
            choices=choices,
            value_type=value_type,
            **kwargs,
        )

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
        self.add_node(kind="Tool", name=name, fidelity=fidelity, **kwargs)

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
        query = """
        MATCH (t:Tool {name: $tool_name}), (v:Variable {name: $variable_name})
        MERGE (t)-[:OUTPUTS]->(v)
        """
        self.graph.query(
            query,
            params={"tool_name": tool_name, "variable_name": variable_name},
        )

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
        query = """
        MATCH (v:Variable {name: $variable_name}), (t:Tool {name: $tool_name})
        MERGE (v)-[:INPUTS_TO]->(t)
        """
        self.graph.query(
            query,
            params={"variable_name": variable_name, "tool_name": tool_name},
        )

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

            vars_list.append(props)
        return vars_list

    def get_tool_inputs(self, tool_name: str) -> list[str]:
        """Retrieves input variables for a specific tool."""
        query = """
        MATCH (v:Variable)-[:INPUTS_TO]->(t:Tool {name: $tool_name})
        RETURN v.name
        """
        result = self.graph.query(query, params={"tool_name": tool_name})
        return [r[0] for r in result.result_set]

    def get_tool_outputs(self, tool_name: str) -> list[str]:
        """Retrieves output variables for a specific tool."""
        query = """
        MATCH (t:Tool {name: $tool_name})-[:OUTPUTS]->(v:Variable)
        RETURN v.name
        """
        result = self.graph.query(query, params={"tool_name": tool_name})
        return [r[0] for r in result.result_set]

    def get_graph_schema(self) -> dict[str, Any]:
        """Returns a serializable dictionary representing the entire graph structure.

        Returns:
            A dictionary containing 'tools' and 'variables' lists defining the topology.

        Example:
            ```python
            schema = gm.get_graph_schema()
            print(schema["tools"][0]["name"])
            ```

        """
        variables = self.get_variables()

        # Single query to get all tools with their inputs and outputs
        query = """
        MATCH (t:Tool)
        OPTIONAL MATCH (v_in:Variable)-[:INPUTS_TO]->(t)
        OPTIONAL MATCH (t)-[:OUTPUTS]->(v_out:Variable)
        RETURN t,
               collect(DISTINCT v_in.name) AS inputs,
               collect(DISTINCT v_out.name) AS outputs
        """
        result = self.graph.query(query)

        tools = []
        for r in result.result_set:
            tool_node = r[0]
            inputs = [name for name in r[1] if name is not None]
            outputs = [name for name in r[2] if name is not None]

            tools.append(
                {
                    "name": tool_node.properties["name"],
                    "fidelity": tool_node.properties.get("fidelity", "high"),
                    "inputs": inputs,
                    "outputs": outputs,
                },
            )

        return {"tools": tools, "variables": variables}
