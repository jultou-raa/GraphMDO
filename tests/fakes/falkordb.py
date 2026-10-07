"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

In-memory stand-in for a FalkorDB graph.

``FakeGraph`` answers exactly the queries of ``GraphManager`` (matched by their
text constants) with real row semantics over in-memory nodes and edges, and
applies FalkorDB's property rules. An unknown query raises ``AssertionError`` so
a test fails when the manager drifts from what the fake implements.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Any

from mdo_framework.db import graph_manager as gm

Row = list[Any]
Params = dict[str, Any]
Handler = Callable[[Params], list[Row]]

INPUTS_TO = "INPUTS_TO"
OUTPUTS = "OUTPUTS"


@dataclass
class FakeNode:
    """Node as stored, and as returned to a caller (a copy)."""

    id: int
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class FakeResult:
    """Query result exposing the rows like the FalkorDB client does."""

    result_set: list[Row]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, bool | int | float | str)


def _stored(props: Params) -> Params:
    """Apply FalkorDB's property rules: null removes, scalars and flat arrays only."""
    stored: Params = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, list):
            flat = all(_is_scalar(item) for item in value)
            if not flat or len({type(item) for item in value}) > 1:
                raise AssertionError(f"FalkorDB cannot store {key}={value!r}")
        elif not _is_scalar(value):
            raise AssertionError(f"FalkorDB cannot store {key}={value!r}")
        stored[key] = deepcopy(value)
    return stored


def _seq_order(node: FakeNode) -> tuple[bool, int]:
    seq = node.properties.get("seq")
    return (seq is None, seq if seq is not None else 0)


class FakeGraph:
    """In-memory graph whose ``query`` mimics the FalkorDB client."""

    def __init__(self) -> None:
        self._nodes: list[FakeNode] = []
        # (relation, variable node id, tool node id)
        self._edges: set[tuple[str, int, int]] = set()
        self._ids = count()
        self._handlers = self._build_handlers()

    def query(self, query: str, params: Params | None = None) -> FakeResult:
        """Run a known query and return its rows."""
        handler = self._handlers.get(query)
        if handler is None:
            raise AssertionError(f"FakeGraph does not implement this query: {query}")
        return FakeResult(handler(params or {}))

    def node(self, label: str, name: str) -> Params | None:
        """Return a copy of the properties of the ``label`` node called ``name``."""
        matches = self._find(label, name)
        if len(matches) > 1:
            raise AssertionError(f"duplicate {label} nodes called {name!r}")
        return deepcopy(matches[0].properties) if matches else None

    def add_raw_node(self, label: str, **properties: Any) -> None:
        """Seed a node directly, as an older version of the code would have."""
        self._add_node(label, properties)

    def add_raw_edge(self, relation: str, variable_name: str, tool_name: str) -> None:
        """Seed an edge directly, bypassing the checks of the manager."""
        (variable,) = self._find("Variable", variable_name)
        (tool,) = self._find("Tool", tool_name)
        self._edges.add((relation, variable.id, tool.id))

    def _build_handlers(self) -> dict[str, Handler]:
        handlers: dict[str, Handler] = {
            gm._CLEAR_GRAPH: self._clear,
            gm._NEXT_SEQ: self._next_seq,
            gm._NODES_BY_NAME: self._nodes_by_name,
            gm._PRODUCERS_OF_VARIABLE: self._producers_of_variable,
            gm._OUTPUT_CHECK: self._output_check,
            gm._IS_OUTPUT_OF_TOOL: self._is_output_of_tool,
            gm._CONNECT_INPUT: partial(self._connect, INPUTS_TO),
            gm._CONNECT_OUTPUT: partial(self._connect, OUTPUTS),
            gm._INPUT_EDGES: partial(self._edge_rows, INPUTS_TO),
            gm._OUTPUT_EDGES: partial(self._edge_rows, OUTPUTS),
        }
        for queries in (gm._VARIABLE_QUERIES, gm._TOOL_QUERIES):
            label = queries.label
            handlers[queries.exists] = partial(self._exists, label)
            handlers[queries.create] = partial(self._create, label)
            handlers[queries.replace] = partial(self._replace, label)
            handlers[queries.delete] = partial(self._delete, label)
            handlers[queries.list_all] = partial(self._list_all, label)
        return handlers

    def _find(self, label: str, name: str) -> list[FakeNode]:
        return [
            node
            for node in self._nodes
            if label in node.labels and node.properties.get("name") == name
        ]

    def _add_node(self, label: str, properties: Params) -> FakeNode:
        node = FakeNode(next(self._ids), [label], _stored(properties))
        self._nodes.append(node)
        return node

    def _by_id(self, node_id: int) -> FakeNode:
        return next(node for node in self._nodes if node.id == node_id)

    def _pairs(self, params: Params) -> list[tuple[FakeNode, FakeNode]]:
        """Every (variable, tool) combination matching the query's names."""
        return [
            (variable, tool)
            for variable in self._find("Variable", params["variable_name"])
            for tool in self._find("Tool", params["tool_name"])
        ]

    def _clear(self, params: Params) -> list[Row]:
        self._nodes.clear()
        self._edges.clear()
        return []

    def _next_seq(self, params: Params) -> list[Row]:
        counters = self._find("Sequence", "seq")
        counter = (
            counters[0]
            if counters
            else self._add_node("Sequence", {"name": "seq", "value": 0})
        )
        counter.properties["value"] += 1
        return [[counter.properties["value"]]]

    def _exists(self, label: str, params: Params) -> list[Row]:
        return [
            [node.properties.get("seq")] for node in self._find(label, params["name"])
        ]

    def _create(self, label: str, params: Params) -> list[Row]:
        self._add_node(label, params["props"])
        return []

    def _replace(self, label: str, params: Params) -> list[Row]:
        for node in self._find(label, params["name"]):
            node.properties = _stored(params["props"])
        return []

    def _delete(self, label: str, params: Params) -> list[Row]:
        for node in self._find(label, params["name"]):
            self._nodes.remove(node)
            self._edges = {edge for edge in self._edges if node.id not in edge[1:]}
        return []

    def _list_all(self, label: str, params: Params) -> list[Row]:
        nodes = sorted(
            (node for node in self._nodes if label in node.labels), key=_seq_order
        )
        return [[deepcopy(node)] for node in nodes]

    def _nodes_by_name(self, params: Params) -> list[Row]:
        return [
            [list(node.labels), node.properties["name"]]
            for node in self._nodes
            if node.properties.get("name") in params["names"]
        ]

    def _producers_of_variable(self, params: Params) -> list[Row]:
        names = {
            self._by_id(tool_id).properties["name"]
            for relation, variable_id, tool_id in self._edges
            if relation == OUTPUTS
            and self._by_id(variable_id).properties.get("name") == params["name"]
        }
        return [[name] for name in sorted(names)]

    def _output_check(self, params: Params) -> list[Row]:
        rows: list[Row] = []
        for variable, tool in self._pairs(params):
            producers = sorted(
                self._by_id(tool_id).properties["name"]
                for relation, variable_id, tool_id in self._edges
                if relation == OUTPUTS and variable_id == variable.id
            )
            is_input = (INPUTS_TO, variable.id, tool.id) in self._edges
            rows.append([variable.properties.get("kind"), producers, int(is_input)])
        return rows

    def _is_output_of_tool(self, params: Params) -> list[Row]:
        found = [
            (OUTPUTS, variable.id, tool.id) in self._edges
            for variable, tool in self._pairs(params)
        ]
        return [[sum(found)]]

    def _connect(self, relation: str, params: Params) -> list[Row]:
        pairs = self._pairs(params)
        self._edges.update((relation, variable.id, tool.id) for variable, tool in pairs)
        return [[len(pairs)]]

    def _edge_rows(self, relation: str, params: Params) -> list[Row]:
        pairs = [
            (self._by_id(variable_id), self._by_id(tool_id))
            for edge_relation, variable_id, tool_id in self._edges
            if edge_relation == relation
        ]
        pairs.sort(key=lambda pair: _seq_order(pair[0]))
        return [
            [tool.properties["name"], variable.properties["name"]]
            for variable, tool in pairs
        ]
