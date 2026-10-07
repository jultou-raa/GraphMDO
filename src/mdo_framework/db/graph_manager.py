"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from mdo_framework.db.client import FalkorDBClient
from mdo_framework.schema import (
    Finding,
    StudySchema,
    StudyValidationError,
    ToolNode,
    ToolSpec,
    ValidationReport,
    Variable,
    report_from_validation_error,
)

_VARIABLE_ADAPTER: TypeAdapter[Variable] = TypeAdapter(Variable)

_VARIABLE = "Variable"
_TOOL = "Tool"
_DESIGN_OR_FIXED_KINDS = ("range", "choice", "fixed")


class NodeNotFoundError(LookupError):
    """Raised when a variable or tool that an operation needs does not exist.

    Attributes:
        missing: ``(label, name)`` pairs, label being ``"Variable"`` or ``"Tool"``.
        hint: Suggestion when the arguments look swapped, else ``None``.
    """

    def __init__(
        self, missing: Iterable[tuple[str, str]], hint: str | None = None
    ) -> None:
        self.missing = tuple(missing)
        self.hint = hint
        names = " and ".join(f"{label} '{name}'" for label, name in self.missing)
        super().__init__(f"{names} not found" + (f". {hint}" if hint else ""))


class NodeExistsError(ValueError):
    """Raised when creating a node whose name is already taken.

    Attributes:
        label: ``"Variable"`` or ``"Tool"``.
        name: Name of the existing node.
    """

    def __init__(self, label: str, name: str) -> None:
        self.label = label
        self.name = name
        super().__init__(f"{label} '{name}' already exists")


class DuplicateProducerError(ValueError):
    """Raised when a second tool would produce an already produced variable.

    Attributes:
        variable: The variable name.
        producers: Tools producing it: the existing ones, then the new one.
    """

    def __init__(self, variable: str, producers: Iterable[str]) -> None:
        self.variable = variable
        self.producers = tuple(producers)
        super().__init__(
            f"Variable '{variable}' would be produced by more than one tool: "
            f"{', '.join(self.producers)}"
        )


class RoleConflictError(ValueError):
    """Raised when a connection or kind change gives a variable two clashing roles.

    Attributes:
        variable: The variable name.
        message: Explanation of the conflict.
    """

    def __init__(self, variable: str, message: str) -> None:
        self.variable = variable
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class _NodeQueries:
    """Cypher statements for one node label."""

    label: str
    exists: str
    add: str
    put: str
    delete: str
    list_all: str


def _node_queries(label: str) -> _NodeQueries:
    # ``label`` is one of two module constants, never user input.
    # ``add`` and ``put`` are single MERGE statements, hence atomic: two writers
    # racing on one name cannot both create a node. ``$seq`` is a number freshly
    # drawn from the counter. ``add`` returns the ``seq`` of the node holding the
    # name, equal to ``$seq`` only if it just created it. ``put`` keeps the old
    # ``seq`` (a legacy node without one gets ``$seq``) and returns whether it
    # created the node, null for that legacy node.
    return _NodeQueries(
        label=label,
        exists=f"MATCH (n:{label} {{name: $name}}) RETURN n.seq",
        add=(
            f"MERGE (n:{label} {{name: $name}}) ON CREATE SET n = $props RETURN n.seq"
        ),
        put=(
            f"MERGE (n:{label} {{name: $name}}) "
            "ON CREATE SET n.seq = $seq "
            "WITH n, n.seq AS old "
            "SET n = $props "
            "SET n.seq = coalesce(old, $seq) "
            "RETURN old = $seq"
        ),
        delete=f"MATCH (n:{label} {{name: $name}}) DETACH DELETE n",
        list_all=f"MATCH (n:{label}) RETURN n ORDER BY n.seq",
    )


_VARIABLE_QUERIES = _node_queries(_VARIABLE)
_TOOL_QUERIES = _node_queries(_TOOL)

_CLEAR_GRAPH = "MATCH (n) DETACH DELETE n"
_NEXT_SEQ = (
    'MERGE (c:Sequence {name: "seq"}) '
    "ON CREATE SET c.value = 0 "
    "SET c.value = c.value + 1 "
    "RETURN c.value"
)
_PRODUCERS_OF_VARIABLE = (
    "MATCH (t:Tool)-[:OUTPUTS]->(v:Variable {name: $name}) RETURN t.name"
)
_OUTPUT_CHECK = (
    "MATCH (t:Tool {name: $tool_name}), (v:Variable {name: $variable_name}) "
    "OPTIONAL MATCH (p:Tool)-[:OUTPUTS]->(v) "
    "OPTIONAL MATCH (v)-[i:INPUTS_TO]->(t) "
    "RETURN v.kind, collect(DISTINCT p.name), count(DISTINCT i)"
)
_IS_OUTPUT_OF_TOOL = (
    "MATCH (:Tool {name: $tool_name})-[o:OUTPUTS]->(:Variable {name: $variable_name}) "
    "RETURN count(o)"
)
_CONNECT_INPUT = (
    "MATCH (v:Variable {name: $variable_name}), (t:Tool {name: $tool_name}) "
    "MERGE (v)-[:INPUTS_TO]->(t) "
    "RETURN count(*)"
)
_CONNECT_OUTPUT = (
    "MATCH (t:Tool {name: $tool_name}), (v:Variable {name: $variable_name}) "
    "MERGE (t)-[:OUTPUTS]->(v) "
    "RETURN count(*)"
)
_NODES_BY_NAME = "MATCH (n) WHERE n.name IN $names RETURN labels(n), n.name"
_INPUT_EDGES = (
    "MATCH (v:Variable)-[:INPUTS_TO]->(t:Tool) RETURN t.name, v.name ORDER BY v.seq"
)
_OUTPUT_EDGES = (
    "MATCH (t:Tool)-[:OUTPUTS]->(v:Variable) RETURN t.name, v.name ORDER BY v.seq"
)


class GraphManager:
    """Typed access to the study graph stored in FalkorDB.

    Variables and tools are created from the models of ``mdo_framework.schema`` and
    read back as models. Each node carries a ``seq`` number, assigned at creation,
    that defines the order of every listing: design variable order is meaningful.
    The numbers may have gaps. Creating or replacing a node is one atomic query, so
    concurrent writers cannot duplicate a name; checks spanning several queries
    (role conflicts, duplicate producers) assume a single writer.

    Attributes:
        graph: The FalkorDB graph that is queried.
    """

    def __init__(self, graph: Any | None = None) -> None:
        """Bind the manager to a graph.

        Args:
            graph: Graph exposing ``query(text, params=...)``. Defaults to the graph
                of the shared ``FalkorDBClient``.
        """
        self.graph = graph if graph is not None else FalkorDBClient().get_graph()

    def clear_graph(self) -> None:
        """Delete every node and edge, and restart the ``seq`` numbering."""
        self._run(_CLEAR_GRAPH)

    # --- Variables ---

    def add_variable(self, variable: Variable) -> None:
        """Create a variable.

        Args:
            variable: The variable to store.

        Raises:
            NodeExistsError: If a variable with this name exists.

        Example:
            ```python
            gm.add_variable(RangeVar(name="x", lower=-10.0, upper=10.0))
            gm.add_variable(StateVar(name="f_xy"))
            ```
        """
        self._create(_VARIABLE_QUERIES, variable)

    def put_variable(self, variable: Variable) -> bool:
        """Create a variable, or replace every property of an existing one.

        The ``seq`` number, hence the position, and the edges are kept on replace.

        Args:
            variable: The variable to store.

        Returns:
            ``True`` if it was created, ``False`` if it replaced an existing one.

        Raises:
            RoleConflictError: If the existing variable is produced by a tool and
                the new kind is not ``"state"``.
        """
        if variable.kind != "state":
            producers = [
                row[0] for row in self._run(_PRODUCERS_OF_VARIABLE, name=variable.name)
            ]
            if producers:
                raise RoleConflictError(
                    variable.name,
                    f"Variable '{variable.name}' is produced by "
                    f"{', '.join(producers)} and must keep kind 'state', "
                    f"not '{variable.kind}'",
                )
        return self._put(_VARIABLE_QUERIES, variable)

    def delete_variable(self, name: str) -> None:
        """Delete a variable and its edges.

        Args:
            name: Variable name.

        Raises:
            NodeNotFoundError: If there is no such variable.
        """
        self._delete(_VARIABLE_QUERIES, name)

    def get_variables(self) -> list[Variable]:
        """List the variables in creation order.

        Returns:
            The stored variables.

        Raises:
            StudyValidationError: With a ``LEGACY_NODE`` finding for every variable
                node created without a ``kind`` by an older version.
        """
        variables: list[Variable] = []
        legacy: list[Finding] = []
        for (node,) in self._run(_VARIABLE_QUERIES.list_all):
            properties = _node_properties(node)
            if "kind" in properties:
                variables.append(_VARIABLE_ADAPTER.validate_python(properties))
                continue
            name = properties.get("name")
            legacy.append(
                Finding(
                    code="LEGACY_NODE",
                    message=(
                        f"Variable '{name}' has no 'kind'; "
                        "delete and recreate it with the typed API"
                    ),
                    names=(name,),
                )
            )
        if legacy:
            raise StudyValidationError(ValidationReport(errors=tuple(legacy)))
        return variables

    # --- Tools ---

    def add_tool(self, tool: ToolNode) -> None:
        """Create a tool.

        Args:
            tool: The tool to store.

        Raises:
            NodeExistsError: If a tool with this name exists.

        Example:
            ```python
            gm.add_tool(ToolNode(name="CFD_Solver", fidelity="high"))
            ```
        """
        self._create(_TOOL_QUERIES, tool)

    def put_tool(self, tool: ToolNode) -> bool:
        """Create a tool, or replace every property of an existing one.

        The ``seq`` number, hence the position, and the edges are kept on replace.

        Args:
            tool: The tool to store.

        Returns:
            ``True`` if it was created, ``False`` if it replaced an existing one.
        """
        return self._put(_TOOL_QUERIES, tool)

    def delete_tool(self, name: str) -> None:
        """Delete a tool and its edges.

        Args:
            name: Tool name.

        Raises:
            NodeNotFoundError: If there is no such tool.
        """
        self._delete(_TOOL_QUERIES, name)

    def get_tools(self) -> list[ToolNode]:
        """List the tools in creation order.

        Returns:
            The stored tools, without their connections.
        """
        return [
            ToolNode.model_validate(_node_properties(node))
            for (node,) in self._run(_TOOL_QUERIES.list_all)
        ]

    # --- Connections ---

    def connect_input_to_tool(self, variable_name: str, tool_name: str) -> None:
        """Connect a variable to a tool that consumes it (Variable -> Tool).

        Connecting twice is harmless.

        Args:
            variable_name: The consumed variable.
            tool_name: The consuming tool.

        Raises:
            NodeNotFoundError: If the variable or the tool does not exist. The hint
                suggests ``connect_tool_to_output`` when the arguments look swapped.
            RoleConflictError: If the variable is already an output of the tool.

        Example:
            ```python
            gm.connect_input_to_tool("x", "Paraboloid")
            ```
        """
        params = {"variable_name": variable_name, "tool_name": tool_name}
        if self._run(_IS_OUTPUT_OF_TOOL, **params)[0][0]:
            raise RoleConflictError(
                variable_name, _both_roles_message(variable_name, tool_name)
            )
        if not self._run(_CONNECT_INPUT, **params)[0][0]:
            raise self._not_found(
                (_VARIABLE, variable_name),
                (_TOOL, tool_name),
                other_direction="connect_tool_to_output(tool_name, variable_name)",
            )

    def connect_tool_to_output(self, tool_name: str, variable_name: str) -> None:
        """Connect a tool to a variable it produces (Tool -> Variable).

        Connecting twice is harmless.

        Args:
            tool_name: The producing tool.
            variable_name: The produced variable, which must be of kind ``"state"``.

        Raises:
            NodeNotFoundError: If the tool or the variable does not exist. The hint
                suggests ``connect_input_to_tool`` when the arguments look swapped.
            DuplicateProducerError: If another tool already produces the variable.
            RoleConflictError: If the variable is a design or fixed variable, or is
                already an input of the tool.

        Example:
            ```python
            gm.connect_tool_to_output("Paraboloid", "f_xy")
            ```
        """
        params = {"tool_name": tool_name, "variable_name": variable_name}
        for kind, producers, input_edges in self._run(_OUTPUT_CHECK, **params):
            others = [producer for producer in producers if producer != tool_name]
            if others:
                raise DuplicateProducerError(variable_name, (*others, tool_name))
            if kind in _DESIGN_OR_FIXED_KINDS:
                raise RoleConflictError(
                    variable_name,
                    f"Variable '{variable_name}' has kind '{kind}', so it cannot be "
                    "a tool output; declare it as kind 'state'",
                )
            if input_edges:
                raise RoleConflictError(
                    variable_name, _both_roles_message(variable_name, tool_name)
                )
        if not self._run(_CONNECT_OUTPUT, **params)[0][0]:
            raise self._not_found(
                (_TOOL, tool_name),
                (_VARIABLE, variable_name),
                other_direction="connect_input_to_tool(variable_name, tool_name)",
            )

    def get_tool_inputs(self, tool_name: str) -> list[str]:
        """List the variables a tool consumes, in variable creation order.

        Args:
            tool_name: Tool name.

        Returns:
            Variable names; empty for an unknown tool.
        """
        return self._ports(_INPUT_EDGES).get(tool_name, [])

    def get_tool_outputs(self, tool_name: str) -> list[str]:
        """List the variables a tool produces, in variable creation order.

        Args:
            tool_name: Tool name.

        Returns:
            Variable names; empty for an unknown tool.
        """
        return self._ports(_OUTPUT_EDGES).get(tool_name, [])

    # --- Study ---

    def get_study_schema(self) -> StudySchema:
        """Read the whole graph as a typed study.

        Returns:
            The variables and the tools with their inputs and outputs, in creation
            order.

        Raises:
            StudyValidationError: If a variable node is a legacy one without a
                ``kind``, or if the stored graph breaks a structural rule.
        """
        variables = self.get_variables()
        inputs = self._ports(_INPUT_EDGES)
        outputs = self._ports(_OUTPUT_EDGES)
        tools = [
            ToolSpec(
                **tool.model_dump(),
                inputs=inputs.get(tool.name, []),
                outputs=outputs.get(tool.name, []),
            )
            for tool in self.get_tools()
        ]
        try:
            return StudySchema(variables=variables, tools=tools)
        except ValidationError as exc:
            raise StudyValidationError(report_from_validation_error(exc)) from exc

    # --- Internals ---

    def _run(self, query: str, **params: Any) -> list[list[Any]]:
        # An empty parameter map would render an invalid ``CYPHER`` header
        return self.graph.query(query, params=params or None).result_set

    def _create(self, queries: _NodeQueries, model: Variable | ToolNode) -> None:
        # The number is drawn first, so a refused creation leaves a gap in ``seq``;
        # only the relative order matters
        seq = self._run(_NEXT_SEQ)[0][0]
        rows = self._run(queries.add, name=model.name, props=_node_props(model, seq))
        if rows[0][0] != seq:
            raise NodeExistsError(queries.label, model.name)

    def _put(self, queries: _NodeQueries, model: Variable | ToolNode) -> bool:
        seq = self._run(_NEXT_SEQ)[0][0]
        rows = self._run(
            queries.put, name=model.name, seq=seq, props=_node_props(model, seq)
        )
        return bool(rows[0][0])

    def _delete(self, queries: _NodeQueries, name: str) -> None:
        if not self._run(queries.exists, name=name):
            raise NodeNotFoundError([(queries.label, name)])
        self._run(queries.delete, name=name)

    def _ports(self, edge_query: str) -> dict[str, list[str]]:
        ports: dict[str, list[str]] = {}
        for tool_name, variable_name in self._run(edge_query):
            ports.setdefault(tool_name, []).append(variable_name)
        return ports

    def _not_found(
        self,
        first: tuple[str, str],
        second: tuple[str, str],
        other_direction: str,
    ) -> NodeNotFoundError:
        """Explain a connection that matched nothing, hinting at swapped arguments."""
        rows = self._run(_NODES_BY_NAME, names=[first[1], second[1]])
        present = {(label, name) for labels, name in rows for label in labels}
        missing = [node for node in (first, second) if node not in present]
        swapped = []
        for label, name in missing:
            actual = _TOOL if label == _VARIABLE else _VARIABLE
            if (actual, name) in present:
                swapped.append(f"'{name}' is a {actual.lower()}")
        hint = None
        if swapped:
            hint = (
                f"{' and '.join(swapped)}; the arguments look swapped, "
                f"use {other_direction} instead"
            )
        return NodeNotFoundError(missing, hint)


def _node_properties(node: Any) -> dict[str, Any]:
    """Return the stored properties of a node without its ``seq`` number."""
    return {key: value for key, value in node.properties.items() if key != "seq"}


def _node_props(model: Variable | ToolNode, seq: int) -> dict[str, Any]:
    return {**model.model_dump(mode="json", exclude_none=True), "seq": seq}


def _both_roles_message(variable_name: str, tool_name: str) -> str:
    return (
        f"Variable '{variable_name}' cannot be both an input and output "
        f"of tool '{tool_name}'"
    )
