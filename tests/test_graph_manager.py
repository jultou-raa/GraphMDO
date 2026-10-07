"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from fakes.falkordb import FakeGraph

from mdo_framework.db import graph_manager
from mdo_framework.db.graph_manager import (
    DuplicateProducerError,
    GraphManager,
    NodeExistsError,
    NodeNotFoundError,
    RoleConflictError,
)
from mdo_framework.schema import (
    ChoiceVar,
    FixedParam,
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolNode,
    ToolSpec,
    Variable,
)


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def manager(graph: FakeGraph) -> GraphManager:
    return GraphManager(graph=graph)


def _range(name: str = "x", **overrides: Any) -> RangeVar:
    return RangeVar(**{"name": name, "lower": -10.0, "upper": 10.0, **overrides})


def _variable_names(manager: GraphManager) -> list[str]:
    return [variable.name for variable in manager.get_variables()]


def _tool_names(manager: GraphManager) -> list[str]:
    return [tool.name for tool in manager.get_tools()]


def _build_paraboloid(manager: GraphManager) -> None:
    manager.add_variable(_range("x"))
    manager.add_variable(_range("y"))
    manager.add_variable(StateVar(name="f_xy"))
    manager.add_variable(StateVar(name="c_xy"))
    manager.add_tool(ToolNode(name="Paraboloid"))
    manager.connect_input_to_tool("x", "Paraboloid")
    manager.connect_input_to_tool("y", "Paraboloid")
    manager.connect_tool_to_output("Paraboloid", "f_xy")
    manager.connect_tool_to_output("Paraboloid", "c_xy")


@pytest.fixture
def paraboloid(manager: GraphManager) -> GraphManager:
    _build_paraboloid(manager)
    return manager


# --- Construction -----------------------------------------------------------


def test_default_graph_comes_from_the_falkordb_client() -> None:
    with patch("mdo_framework.db.graph_manager.FalkorDBClient") as client_cls:
        manager = GraphManager()
    assert manager.graph is client_cls.return_value.get_graph.return_value


def test_injected_graph_is_used_without_touching_the_client(
    graph: FakeGraph,
) -> None:
    with patch("mdo_framework.db.graph_manager.FalkorDBClient") as client_cls:
        manager = GraphManager(graph=graph)
    assert manager.graph is graph
    client_cls.assert_not_called()


def test_fake_graph_rejects_a_query_it_does_not_implement(graph: FakeGraph) -> None:
    with pytest.raises(AssertionError, match="does not implement"):
        graph.query("MATCH (n) RETURN n")


def _seed_two_producers_and_an_input(graph: FakeGraph) -> None:
    graph.add_raw_node("Variable", name="v", kind="state")
    for tool in ("A", "B", "T"):
        graph.add_raw_node("Tool", name=tool)
    graph.add_raw_edge("OUTPUTS", "v", "A")
    graph.add_raw_edge("OUTPUTS", "v", "B")
    graph.add_raw_edge("INPUTS_TO", "v", "T")


def _output_check(graph: FakeGraph, tool: str) -> list[list[Any]]:
    params = {"tool_name": tool, "variable_name": "v"}
    return graph.query(graph_manager._OUTPUT_CHECK, params=params).result_set


def test_fake_output_check_counts_one_input_edge_across_two_producers(
    graph: FakeGraph,
) -> None:
    _seed_two_producers_and_an_input(graph)
    # two producer rows times one input row must not be counted twice
    assert _output_check(graph, "T") == [["state", ["A", "B"], 1]]


def test_fake_output_check_ignores_the_null_rows_of_the_optional_matches(
    graph: FakeGraph,
) -> None:
    _seed_two_producers_and_an_input(graph)
    assert _output_check(graph, "A") == [["state", ["A", "B"], 0]]
    graph.add_raw_node("Variable", name="w", kind="range")
    params = {"tool_name": "T", "variable_name": "w"}
    result = graph.query(graph_manager._OUTPUT_CHECK, params=params)
    assert result.result_set == [["range", [], 0]]


def test_fake_output_check_returns_no_row_for_an_unknown_node(
    graph: FakeGraph,
) -> None:
    _seed_two_producers_and_an_input(graph)
    assert _output_check(graph, "ghost") == []


class _RacingGraph(FakeGraph):
    """Graph where a second writer acts right after the first draws its ``seq``."""

    rival: Callable[[], Any] | None = None

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        result = super().query(query, params)
        if query == graph_manager._NEXT_SEQ and self.rival is not None:
            rival, self.rival = self.rival, None
            rival()
        return result


# --- Variables --------------------------------------------------------------


def test_add_variable_stores_the_model_and_a_sequence_number(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_variable(_range("x", initial=1.0))
    assert graph.node("Variable", "x") == {
        "kind": "range",
        "name": "x",
        "lower": -10.0,
        "upper": 10.0,
        "value_type": "float",
        "initial": 1.0,
        "seq": 1,
    }


def test_add_variable_omits_unset_optional_fields(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_variable(StateVar(name="f"))
    assert graph.node("Variable", "f") == {"kind": "state", "name": "f", "seq": 1}


def test_add_variable_rejects_an_existing_name(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_variable(_range("x"))
    with pytest.raises(NodeExistsError) as exc_info:
        manager.add_variable(FixedParam(name="x", value=1))
    assert (exc_info.value.label, exc_info.value.name) == ("Variable", "x")
    assert "Variable 'x'" in str(exc_info.value)
    assert manager.get_variables() == [_range("x")]
    assert graph.node("Variable", "x")["seq"] == 1


@pytest.mark.parametrize(
    "variable",
    [
        _range("x", initial=2.5, units="m"),
        _range("n", lower=0.0, upper=5.0, value_type="int", initial=3.0),
        ChoiceVar(name="material", choices=["steel", "alu"], initial="alu"),
        ChoiceVar(name="count", choices=[1, 2, 3]),
        ChoiceVar(name="ratio", choices=[0.5, 1.5]),
        ChoiceVar(name="flag", choices=[True, False]),
        FixedParam(name="label", value="abc", units="-"),
        FixedParam(name="n_blades", value=3),
        FixedParam(name="gain", value=0.25),
        FixedParam(name="enabled", value=True),
        StateVar(name="f"),
        StateVar(name="g", initial_guess=1.5),
        StateVar(name="h", initial_guess=[1.0, 2.0, 3.0]),
    ],
    ids=lambda variable: variable.name,
)
def test_every_variable_shape_survives_a_round_trip(
    manager: GraphManager, variable: Variable
) -> None:
    manager.add_variable(variable)
    (restored,) = manager.get_variables()
    assert restored == variable
    # JSON keeps ``True`` and ``1`` apart, which model equality does not
    assert restored.model_dump_json() == variable.model_dump_json()


def test_put_variable_creates_then_replaces(manager: GraphManager) -> None:
    assert manager.put_variable(_range("x")) is True
    assert manager.put_variable(_range("x", upper=20.0)) is False
    assert manager.get_variables() == [_range("x", upper=20.0)]


def test_put_variable_replaces_every_property_and_keeps_the_sequence_number(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_variable(_range("x", initial=1.0, units="m"))
    manager.add_variable(_range("y"))
    manager.put_variable(FixedParam(name="x", value=2))
    assert graph.node("Variable", "x") == {
        "kind": "fixed",
        "name": "x",
        "value": 2,
        "seq": 1,
    }
    assert _variable_names(manager) == ["x", "y"]


def test_put_variable_keeps_a_produced_variable_as_state(
    paraboloid: GraphManager,
) -> None:
    assert paraboloid.put_variable(StateVar(name="f_xy", initial_guess=2.0)) is False
    schema = paraboloid.get_study_schema()
    assert schema.tool("Paraboloid").outputs == ["f_xy", "c_xy"]
    assert schema.variable("f_xy") == StateVar(name="f_xy", initial_guess=2.0)


@pytest.mark.parametrize(
    "variable",
    [
        _range("f_xy"),
        ChoiceVar(name="f_xy", choices=[1, 2]),
        FixedParam(name="f_xy", value=1.0),
    ],
    ids=lambda variable: variable.kind,
)
def test_put_variable_refuses_to_turn_a_produced_variable_into_an_input(
    paraboloid: GraphManager, variable: Variable
) -> None:
    with pytest.raises(RoleConflictError) as exc_info:
        paraboloid.put_variable(variable)
    assert exc_info.value.variable == "f_xy"
    assert "Paraboloid" in exc_info.value.message
    assert str(exc_info.value) == exc_info.value.message
    assert paraboloid.get_study_schema().variable("f_xy") == StateVar(name="f_xy")


def test_put_variable_allows_any_kind_on_an_unproduced_variable(
    manager: GraphManager,
) -> None:
    manager.add_variable(StateVar(name="f"))
    assert manager.put_variable(FixedParam(name="f", value=1)) is False


def test_delete_variable_removes_the_node_and_its_edges(
    paraboloid: GraphManager, graph: FakeGraph
) -> None:
    paraboloid.delete_variable("x")
    paraboloid.delete_variable("f_xy")
    assert graph.node("Variable", "x") is None
    assert _variable_names(paraboloid) == ["y", "c_xy"]
    tool = paraboloid.get_study_schema().tool("Paraboloid")
    assert (tool.inputs, tool.outputs) == (["y"], ["c_xy"])


def test_delete_variable_reports_a_missing_variable(manager: GraphManager) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        manager.delete_variable("ghost")
    assert exc_info.value.missing == (("Variable", "ghost"),)
    assert exc_info.value.hint is None
    assert "Variable 'ghost'" in str(exc_info.value)


# --- Tools ------------------------------------------------------------------


def test_add_tool_stores_the_model_and_a_sequence_number(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_tool(ToolNode(name="Solver", fidelity="low"))
    assert graph.node("Tool", "Solver") == {
        "name": "Solver",
        "fidelity": "low",
        "deterministic": True,
        "arg_map": "{}",
        "seq": 1,
    }
    assert manager.get_tools() == [ToolNode(name="Solver", fidelity="low")]


def test_tool_options_are_stored_and_arg_map_is_a_json_string(
    manager: GraphManager, graph: FakeGraph
) -> None:
    tool = ToolNode(
        name="Solver", deterministic=False, arg_map={"x": "a", "alpha": "b"}
    )
    manager.add_tool(tool)
    properties = graph.node("Tool", "Solver")
    assert properties["deterministic"] is False
    assert properties["arg_map"] == '{"x": "a", "alpha": "b"}'
    assert manager.get_tools() == [tool]


def test_tool_options_round_trip_through_the_study_schema(
    paraboloid: GraphManager,
) -> None:
    paraboloid.put_tool(
        ToolNode(name="Paraboloid", deterministic=False, arg_map={"x": "a"})
    )
    tool = paraboloid.get_study_schema().tool("Paraboloid")
    assert tool.deterministic is False
    assert tool.arg_map == {"x": "a"}
    assert (tool.inputs, tool.outputs) == (["x", "y"], ["f_xy", "c_xy"])


def test_a_tool_node_stored_without_options_gets_the_defaults(
    manager: GraphManager, graph: FakeGraph
) -> None:
    graph.add_raw_node("Tool", name="Old", fidelity="low", seq=1)
    assert manager.get_tools() == [ToolNode(name="Old", fidelity="low")]


def test_add_tool_rejects_an_existing_name(manager: GraphManager) -> None:
    manager.add_tool(ToolNode(name="Solver"))
    with pytest.raises(NodeExistsError) as exc_info:
        manager.add_tool(ToolNode(name="Solver", fidelity="low"))
    assert (exc_info.value.label, exc_info.value.name) == ("Tool", "Solver")
    assert "Tool 'Solver'" in str(exc_info.value)
    assert manager.get_tools() == [ToolNode(name="Solver")]


def test_put_tool_creates_then_replaces_and_keeps_the_sequence_number(
    manager: GraphManager, graph: FakeGraph
) -> None:
    assert manager.put_tool(ToolNode(name="A")) is True
    manager.add_tool(ToolNode(name="B"))
    assert manager.put_tool(ToolNode(name="A", fidelity="low")) is False
    assert manager.get_tools() == [
        ToolNode(name="A", fidelity="low"),
        ToolNode(name="B"),
    ]
    assert graph.node("Tool", "A")["seq"] == 1


def test_put_tool_keeps_the_tool_edges(paraboloid: GraphManager) -> None:
    paraboloid.put_tool(ToolNode(name="Paraboloid", fidelity="low"))
    tool = paraboloid.get_study_schema().tool("Paraboloid")
    assert (tool.fidelity, tool.inputs, tool.outputs) == (
        "low",
        ["x", "y"],
        ["f_xy", "c_xy"],
    )


def test_delete_tool_removes_the_node_and_its_edges_but_not_the_variables(
    paraboloid: GraphManager, graph: FakeGraph
) -> None:
    paraboloid.delete_tool("Paraboloid")
    assert graph.node("Tool", "Paraboloid") is None
    assert paraboloid.get_tools() == []
    assert _variable_names(paraboloid) == ["x", "y", "f_xy", "c_xy"]
    # the outputs are free again
    paraboloid.add_tool(ToolNode(name="Other"))
    paraboloid.connect_tool_to_output("Other", "f_xy")


def test_delete_tool_reports_a_missing_tool(manager: GraphManager) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        manager.delete_tool("ghost")
    assert exc_info.value.missing == (("Tool", "ghost"),)
    assert exc_info.value.hint is None


# --- Clearing ---------------------------------------------------------------


def test_clear_graph_removes_everything_and_restarts_the_sequence(
    paraboloid: GraphManager, graph: FakeGraph
) -> None:
    paraboloid.clear_graph()
    assert paraboloid.get_study_schema() == StudySchema()
    paraboloid.add_variable(_range("z"))
    assert graph.node("Variable", "z")["seq"] == 1


# --- Connections ------------------------------------------------------------


def test_connect_input_to_tool_links_the_variable(manager: GraphManager) -> None:
    manager.add_variable(_range("x"))
    manager.add_tool(ToolNode(name="T"))
    manager.connect_input_to_tool("x", "T")
    assert manager.get_tool_inputs("T") == ["x"]
    assert manager.get_tool_outputs("T") == []


def test_connect_tool_to_output_links_the_variable(manager: GraphManager) -> None:
    manager.add_variable(StateVar(name="f"))
    manager.add_tool(ToolNode(name="T"))
    manager.connect_tool_to_output("T", "f")
    assert manager.get_tool_outputs("T") == ["f"]
    assert manager.get_tool_inputs("T") == []


def test_reconnecting_the_same_edge_is_idempotent(paraboloid: GraphManager) -> None:
    paraboloid.connect_input_to_tool("x", "Paraboloid")
    paraboloid.connect_tool_to_output("Paraboloid", "f_xy")
    assert paraboloid.get_tool_inputs("Paraboloid") == ["x", "y"]
    assert paraboloid.get_tool_outputs("Paraboloid") == ["f_xy", "c_xy"]


def test_a_state_variable_can_feed_another_tool(paraboloid: GraphManager) -> None:
    paraboloid.add_tool(ToolNode(name="Next"))
    paraboloid.connect_input_to_tool("f_xy", "Next")
    assert paraboloid.get_tool_inputs("Next") == ["f_xy"]


@pytest.mark.parametrize(
    ("connect", "arguments", "missing"),
    [
        ("connect_input_to_tool", ("ghost", "Paraboloid"), (("Variable", "ghost"),)),
        ("connect_input_to_tool", ("x", "ghost"), (("Tool", "ghost"),)),
        (
            "connect_input_to_tool",
            ("ghost_v", "ghost_t"),
            (("Variable", "ghost_v"), ("Tool", "ghost_t")),
        ),
        ("connect_tool_to_output", ("Paraboloid", "ghost"), (("Variable", "ghost"),)),
        ("connect_tool_to_output", ("ghost", "f_xy"), (("Tool", "ghost"),)),
        (
            "connect_tool_to_output",
            ("ghost_t", "ghost_v"),
            (("Tool", "ghost_t"), ("Variable", "ghost_v")),
        ),
    ],
    ids=[
        "input-missing-variable",
        "input-missing-tool",
        "input-both-missing",
        "output-missing-variable",
        "output-missing-tool",
        "output-both-missing",
    ],
)
def test_connecting_a_missing_node_names_it_and_changes_nothing(
    paraboloid: GraphManager,
    connect: str,
    arguments: tuple[str, str],
    missing: tuple[tuple[str, str], ...],
) -> None:
    before = paraboloid.get_study_schema()
    with pytest.raises(NodeNotFoundError) as exc_info:
        getattr(paraboloid, connect)(*arguments)
    assert exc_info.value.missing == missing
    assert exc_info.value.hint is None
    for label, name in missing:
        assert f"{label} '{name}'" in str(exc_info.value)
    assert paraboloid.get_study_schema() == before


def test_a_missing_tool_is_reported_before_a_role_conflict(
    paraboloid: GraphManager,
) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        paraboloid.connect_tool_to_output("ghost", "x")
    assert exc_info.value.missing == (("Tool", "ghost"),)


def test_swapped_input_connection_hints_at_the_output_direction(
    paraboloid: GraphManager,
) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        paraboloid.connect_input_to_tool("Paraboloid", "f_xy")
    assert exc_info.value.missing == (
        ("Variable", "Paraboloid"),
        ("Tool", "f_xy"),
    )
    assert exc_info.value.hint is not None
    assert "connect_tool_to_output" in exc_info.value.hint
    assert "'Paraboloid' is a tool" in exc_info.value.hint
    assert exc_info.value.hint in str(exc_info.value)


def test_swapped_output_connection_hints_at_the_input_direction(
    paraboloid: GraphManager,
) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        paraboloid.connect_tool_to_output("x", "Paraboloid")
    assert exc_info.value.missing == (("Tool", "x"), ("Variable", "Paraboloid"))
    assert exc_info.value.hint is not None
    assert "connect_input_to_tool" in exc_info.value.hint
    assert "'x' is a variable" in exc_info.value.hint


def test_a_half_swapped_connection_still_hints(paraboloid: GraphManager) -> None:
    with pytest.raises(NodeNotFoundError) as exc_info:
        paraboloid.connect_input_to_tool("x", "y")
    assert exc_info.value.missing == (("Tool", "y"),)
    assert exc_info.value.hint is not None
    assert "'y' is a variable" in exc_info.value.hint
    assert "connect_tool_to_output" in exc_info.value.hint

    with pytest.raises(NodeNotFoundError) as exc_info:
        paraboloid.connect_input_to_tool("Paraboloid", "Paraboloid")
    assert exc_info.value.missing == (("Variable", "Paraboloid"),)
    assert exc_info.value.hint is not None
    assert "'Paraboloid' is a tool" in exc_info.value.hint


def test_two_producers_of_one_variable_are_rejected(
    paraboloid: GraphManager,
) -> None:
    paraboloid.add_tool(ToolNode(name="Other"))
    with pytest.raises(DuplicateProducerError) as exc_info:
        paraboloid.connect_tool_to_output("Other", "f_xy")
    assert exc_info.value.variable == "f_xy"
    assert exc_info.value.producers == ("Paraboloid", "Other")
    assert "'f_xy'" in str(exc_info.value)
    assert "Paraboloid" in str(exc_info.value)
    assert "Other" in str(exc_info.value)
    assert paraboloid.get_tool_outputs("Other") == []


@pytest.mark.parametrize(
    "variable",
    [
        _range("d"),
        ChoiceVar(name="d", choices=["a", "b"]),
        FixedParam(name="d", value=1.0),
    ],
    ids=lambda variable: variable.kind,
)
def test_a_design_or_fixed_variable_cannot_be_a_tool_output(
    paraboloid: GraphManager, variable: Variable
) -> None:
    paraboloid.add_variable(variable)
    with pytest.raises(RoleConflictError) as exc_info:
        paraboloid.connect_tool_to_output("Paraboloid", "d")
    assert exc_info.value.variable == "d"
    assert "kind 'state'" in exc_info.value.message
    assert paraboloid.get_tool_outputs("Paraboloid") == ["f_xy", "c_xy"]


def test_an_input_of_a_tool_cannot_also_be_its_output(
    paraboloid: GraphManager,
) -> None:
    paraboloid.add_variable(StateVar(name="s"))
    paraboloid.connect_input_to_tool("s", "Paraboloid")
    with pytest.raises(RoleConflictError) as exc_info:
        paraboloid.connect_tool_to_output("Paraboloid", "s")
    assert exc_info.value.variable == "s"
    assert "input and output" in exc_info.value.message
    assert paraboloid.get_tool_outputs("Paraboloid") == ["f_xy", "c_xy"]


def test_an_output_of_a_tool_cannot_also_be_its_input(
    paraboloid: GraphManager,
) -> None:
    with pytest.raises(RoleConflictError) as exc_info:
        paraboloid.connect_input_to_tool("f_xy", "Paraboloid")
    assert exc_info.value.variable == "f_xy"
    assert "input and output" in exc_info.value.message
    assert paraboloid.get_tool_inputs("Paraboloid") == ["x", "y"]


# --- Concurrent writers -----------------------------------------------------


def _racing_pair() -> tuple[_RacingGraph, GraphManager, GraphManager]:
    racing = _RacingGraph()
    return racing, GraphManager(graph=racing), GraphManager(graph=racing)


def test_add_variable_loses_a_race_without_duplicating_the_node() -> None:
    racing, manager, rival = _racing_pair()
    racing.rival = lambda: rival.add_variable(_range("x", upper=1.0))
    with pytest.raises(NodeExistsError):
        manager.add_variable(_range("x"))
    # the winner is untouched; ``node`` fails on duplicate nodes
    assert racing.node("Variable", "x")["upper"] == 1.0
    assert manager.get_variables() == [_range("x", upper=1.0)]


def test_add_tool_loses_a_race_without_duplicating_the_node() -> None:
    racing, manager, rival = _racing_pair()
    racing.rival = lambda: rival.add_tool(ToolNode(name="T", fidelity="low"))
    with pytest.raises(NodeExistsError):
        manager.add_tool(ToolNode(name="T"))
    assert racing.node("Tool", "T")["fidelity"] == "low"
    assert manager.get_tools() == [ToolNode(name="T", fidelity="low")]


def test_put_variable_racing_a_creation_replaces_it_and_keeps_its_position() -> None:
    racing, manager, rival = _racing_pair()
    racing.rival = lambda: rival.add_variable(_range("x", upper=1.0))
    assert manager.put_variable(_range("x", upper=2.0)) is False
    assert racing.node("Variable", "x")["upper"] == 2.0
    assert _variable_names(manager) == ["x"]
    manager.add_variable(_range("y"))
    assert _variable_names(manager) == ["x", "y"]


def test_put_tool_racing_a_creation_replaces_it_and_keeps_its_position() -> None:
    racing, manager, rival = _racing_pair()
    racing.rival = lambda: rival.add_tool(ToolNode(name="T"))
    assert manager.put_tool(ToolNode(name="T", fidelity="low")) is False
    assert racing.node("Tool", "T")["fidelity"] == "low"
    assert _tool_names(manager) == ["T"]


def test_refused_and_replacing_writes_leave_the_order_unchanged(
    manager: GraphManager,
) -> None:
    manager.add_variable(_range("x"))
    with pytest.raises(NodeExistsError):
        manager.add_variable(_range("x"))
    manager.put_variable(_range("x", upper=1.0))
    manager.add_variable(_range("y"))
    manager.put_variable(_range("x", upper=2.0))
    manager.put_variable(_range("z"))
    assert _variable_names(manager) == ["x", "y", "z"]


def test_put_variable_numbers_a_legacy_node_that_has_no_sequence_number(
    manager: GraphManager, graph: FakeGraph
) -> None:
    manager.add_variable(_range("a"))
    graph.add_raw_node("Variable", name="old_x", param_type="continuous")
    assert manager.put_variable(_range("old_x")) is False
    assert graph.node("Variable", "old_x")["seq"] > graph.node("Variable", "a")["seq"]
    assert manager.get_variables() == [_range("a"), _range("old_x")]


# --- Ordering ---------------------------------------------------------------


def test_variables_keep_insertion_order_and_a_recreated_one_goes_last(
    manager: GraphManager,
) -> None:
    for name in ("z", "a", "m"):
        manager.add_variable(_range(name))
    assert _variable_names(manager) == ["z", "a", "m"]

    manager.delete_variable("a")
    manager.add_variable(_range("a"))
    assert _variable_names(manager) == ["z", "m", "a"]

    manager.put_variable(_range("z", upper=1.0))
    assert _variable_names(manager) == ["z", "m", "a"]


def test_tools_keep_insertion_order_and_a_recreated_one_goes_last(
    manager: GraphManager,
) -> None:
    for name in ("Zeta", "Alpha", "Mid"):
        manager.add_tool(ToolNode(name=name))
    manager.delete_tool("Alpha")
    manager.add_tool(ToolNode(name="Alpha"))
    assert _tool_names(manager) == ["Zeta", "Mid", "Alpha"]


def test_tool_ports_follow_the_variable_order_not_the_connection_order(
    manager: GraphManager,
) -> None:
    for name in ("b", "a", "c"):
        manager.add_variable(_range(name))
    for name in ("g", "f"):
        manager.add_variable(StateVar(name=name))
    manager.add_tool(ToolNode(name="T"))
    for name in ("c", "a", "b"):
        manager.connect_input_to_tool(name, "T")
    for name in ("f", "g"):
        manager.connect_tool_to_output("T", name)
    assert manager.get_tool_inputs("T") == ["b", "a", "c"]
    assert manager.get_tool_outputs("T") == ["g", "f"]


# --- Study schema -----------------------------------------------------------


def test_empty_graph_gives_an_empty_study(manager: GraphManager) -> None:
    assert manager.get_study_schema() == StudySchema()


def test_study_schema_round_trips_the_stored_graph(manager: GraphManager) -> None:
    variables: list[Variable] = [
        _range("x", initial=1.0),
        ChoiceVar(name="material", choices=["steel", "alu"]),
        FixedParam(name="gain", value=2),
        StateVar(name="y", initial_guess=[0.5, 1.5]),
        StateVar(name="f"),
    ]
    for variable in variables:
        manager.add_variable(variable)
    for tool in (ToolNode(name="Low", fidelity="low"), ToolNode(name="High")):
        manager.add_tool(tool)
    for variable, tool in (("x", "Low"), ("material", "Low"), ("gain", "Low")):
        manager.connect_input_to_tool(variable, tool)
    manager.connect_tool_to_output("Low", "y")
    manager.connect_input_to_tool("x", "High")
    manager.connect_input_to_tool("y", "High")
    manager.connect_tool_to_output("High", "f")

    assert manager.get_study_schema() == StudySchema(
        variables=variables,
        tools=[
            ToolSpec(
                name="Low",
                fidelity="low",
                inputs=["x", "material", "gain"],
                outputs=["y"],
            ),
            ToolSpec(name="High", inputs=["x", "y"], outputs=["f"]),
        ],
    )


def test_study_schema_of_the_demo_graph(paraboloid: GraphManager) -> None:
    assert paraboloid.get_study_schema() == StudySchema(
        variables=[
            _range("x"),
            _range("y"),
            StateVar(name="f_xy"),
            StateVar(name="c_xy"),
        ],
        tools=[
            ToolSpec(
                name="Paraboloid",
                inputs=["x", "y"],
                outputs=["f_xy", "c_xy"],
            )
        ],
    )


def test_a_stored_graph_that_breaks_a_structural_rule_is_reported(
    paraboloid: GraphManager, graph: FakeGraph
) -> None:
    paraboloid.add_tool(ToolNode(name="Other"))
    graph.add_raw_edge("OUTPUTS", "f_xy", "Other")
    with pytest.raises(StudyValidationError) as exc_info:
        paraboloid.get_study_schema()
    assert [finding.code for finding in exc_info.value.report.errors] == [
        "DUPLICATE_PRODUCER"
    ]


# --- Legacy nodes -----------------------------------------------------------


@pytest.fixture(params=["get_variables", "get_study_schema"])
def read_graph(
    request: pytest.FixtureRequest, manager: GraphManager
) -> Callable[[], Any]:
    return getattr(manager, request.param)


def test_variable_nodes_without_kind_are_all_reported(
    manager: GraphManager, graph: FakeGraph, read_graph: Callable[[], Any]
) -> None:
    graph.add_raw_node("Variable", name="old_x", param_type="continuous", seq=1)
    manager.add_variable(_range("typed"))
    graph.add_raw_node("Variable", name="old_y", value=1.0)
    with pytest.raises(StudyValidationError) as exc_info:
        read_graph()
    errors = exc_info.value.report.errors
    assert [(finding.code, finding.names) for finding in errors] == [
        ("LEGACY_NODE", ("old_x",)),
        ("LEGACY_NODE", ("old_y",)),
    ]
    assert errors[0].message == (
        "Variable 'old_x' has no 'kind'; delete and recreate it with the typed API"
    )
    assert "old_y" in str(exc_info.value)


def test_legacy_variable_can_be_deleted_and_recreated(
    manager: GraphManager, graph: FakeGraph
) -> None:
    graph.add_raw_node("Variable", name="old_x", param_type="continuous")
    manager.delete_variable("old_x")
    manager.add_variable(_range("old_x"))
    assert manager.get_variables() == [_range("old_x")]
