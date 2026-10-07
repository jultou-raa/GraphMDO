"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from mdo_framework.core.dependencies import DependencyWalk, walk_dependencies
from mdo_framework.schema import (
    ChoiceVar,
    FixedParam,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
)


def _range(name: str) -> RangeVar:
    return RangeVar(name=name, lower=0.0, upper=1.0)


def _tool(name: str, inputs: list[str], outputs: list[str]) -> ToolSpec:
    return ToolSpec(name=name, inputs=inputs, outputs=outputs)


def _names(items: tuple) -> tuple[str, ...]:
    return tuple(item.name for item in items)


def _sellar() -> StudySchema:
    return StudySchema(
        variables=[
            _range("x1"),
            _range("z"),
            StateVar(name="y1"),
            StateVar(name="y2"),
            StateVar(name="f"),
        ],
        tools=[
            _tool("disc1", ["z", "x1", "y2"], ["y1"]),
            _tool("disc2", ["z", "y1"], ["y2"]),
            _tool("objective", ["x1", "z", "y1", "y2"], ["f"]),
        ],
    )


def test_chain_through_two_tools() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="a"),
            StateVar(name="b"),
        ],
        tools=[_tool("t1", ["x"], ["a"]), _tool("t2", ["a"], ["b"])],
    )

    walk = walk_dependencies(schema, ["b"])

    assert isinstance(walk, DependencyWalk)
    assert _names(walk.design_variables) == ("x",)
    assert walk.fixed_parameters == ()
    assert walk.tools == ("t1", "t2")
    assert walk.unknown_targets == ()
    assert walk.unproduced_targets == ()
    assert walk.unproduced_states == ()
    assert walk.couplings == ()


def test_target_stops_the_walk_at_its_producer() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="a"),
            StateVar(name="b"),
        ],
        tools=[_tool("t1", ["x"], ["a"]), _tool("t2", ["a"], ["b"])],
    )

    walk = walk_dependencies(schema, ["a"])

    assert walk.tools == ("t1",)


def test_coupled_pair_reports_couplings_in_schema_order() -> None:
    walk = walk_dependencies(_sellar(), ["f"])

    assert walk.couplings == ("y1", "y2")
    assert walk.tools == ("disc1", "disc2", "objective")
    assert _names(walk.design_variables) == ("x1", "z")


def test_output_only_consumed_outside_the_cycle_is_not_a_coupling() -> None:
    walk = walk_dependencies(_sellar(), ["f"])

    assert "f" not in walk.couplings


def test_three_tool_cycle_couples_every_exchanged_variable() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="a"),
            StateVar(name="b"),
            StateVar(name="c"),
        ],
        tools=[
            _tool("ta", ["x", "c"], ["a"]),
            _tool("tb", ["a"], ["b"]),
            _tool("tc", ["b"], ["c"]),
        ],
    )

    walk = walk_dependencies(schema, ["c"])

    assert walk.couplings == ("a", "b", "c")
    assert walk.tools == ("ta", "tb", "tc")


def test_walk_terminates_when_the_target_is_inside_a_cycle() -> None:
    walk = walk_dependencies(_sellar(), ["y1"])

    assert walk.tools == ("disc1", "disc2")
    assert walk.couplings == ("y1", "y2")


def test_couplings_cover_the_whole_schema_not_only_required_tools() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="f"),
            StateVar(name="p"),
            StateVar(name="q"),
        ],
        tools=[
            _tool("main", ["x"], ["f"]),
            _tool("loop_p", ["q"], ["p"]),
            _tool("loop_q", ["p"], ["q"]),
        ],
    )

    walk = walk_dependencies(schema, ["f"])

    assert walk.tools == ("main",)
    assert walk.couplings == ("p", "q")


def test_fixed_parameters_are_separated_from_design_variables() -> None:
    schema = StudySchema(
        variables=[
            FixedParam(name="rho", value=1.2),
            _range("x"),
            ChoiceVar(name="mode", choices=["a", "b"]),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "rho", "mode"], ["f"])],
    )

    walk = walk_dependencies(schema, ["f"])

    assert _names(walk.design_variables) == ("x", "mode")
    assert _names(walk.fixed_parameters) == ("rho",)
    assert isinstance(walk.design_variables[1], ChoiceVar)


def test_unknown_and_unproduced_targets_follow_target_order() -> None:
    schema = StudySchema(
        variables=[_range("x"), StateVar(name="orphan"), StateVar(name="f")],
        tools=[_tool("t", ["x"], ["f"])],
    )

    walk = walk_dependencies(schema, ["zzz", "orphan", "f", "aaa", "x"])

    assert walk.unknown_targets == ("zzz", "aaa")
    assert walk.unproduced_targets == ("orphan", "x")
    assert walk.tools == ("t",)
    assert _names(walk.design_variables) == ("x",)


def test_unproduced_state_input_is_reported() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="s_late"),
            StateVar(name="s_early"),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "s_early", "s_late"], ["f"])],
    )

    walk = walk_dependencies(schema, ["f"])

    assert walk.unproduced_states == ("s_late", "s_early")
    assert _names(walk.design_variables) == ("x",)
    assert walk.tools == ("t",)


def test_order_follows_the_schema_not_traversal_or_alphabet() -> None:
    schema = StudySchema(
        variables=[
            _range("zeta"),
            FixedParam(name="pi_2", value=3.14),
            _range("alpha"),
            FixedParam(name="e_2", value=2.71),
            _range("mach"),
            StateVar(name="u"),
            StateVar(name="v"),
            StateVar(name="f"),
        ],
        tools=[
            _tool("tool_b", ["mach", "zeta", "alpha", "e_2"], ["u"]),
            _tool("tool_c", ["u", "pi_2"], ["v"]),
            _tool("tool_a", ["v"], ["f"]),
        ],
    )

    walk = walk_dependencies(schema, ["f"])

    assert _names(walk.design_variables) == ("zeta", "alpha", "mach")
    assert _names(walk.fixed_parameters) == ("pi_2", "e_2")
    assert walk.tools == ("tool_b", "tool_c", "tool_a")


def test_unrequired_tools_and_variables_are_excluded() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            _range("unused"),
            StateVar(name="f"),
            StateVar(name="g"),
        ],
        tools=[_tool("needed", ["x"], ["f"]), _tool("extra", ["unused"], ["g"])],
    )

    walk = walk_dependencies(schema, ["f"])

    assert walk.tools == ("needed",)
    assert _names(walk.design_variables) == ("x",)


def test_shared_inputs_and_repeated_targets_are_not_duplicated() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="a"),
            StateVar(name="b"),
            StateVar(name="f"),
        ],
        tools=[
            _tool("t1", ["x"], ["a"]),
            _tool("t2", ["x"], ["b"]),
            _tool("t3", ["a", "b"], ["f"]),
        ],
    )

    walk = walk_dependencies(schema, ["f", "f", "unknown", "unknown"])

    assert _names(walk.design_variables) == ("x",)
    assert walk.tools == ("t1", "t2", "t3")
    assert walk.unknown_targets == ("unknown",)


def test_empty_targets_require_nothing() -> None:
    walk = walk_dependencies(_sellar(), [])

    assert walk.design_variables == ()
    assert walk.tools == ()
    assert walk.couplings == ("y1", "y2")
