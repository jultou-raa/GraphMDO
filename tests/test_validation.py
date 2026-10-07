"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from mdo_framework import validation
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    Finding,
    FixedParam,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
    ValidationReport,
)
from mdo_framework.validation import (
    LinearConstraint,
    ParameterConstraintError,
    check_tool_signature,
    parse_parameter_constraints,
    validate_study,
)


@pytest.fixture
def design_space_class() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gemseo.algos.design_space import DesignSpace
    return DesignSpace


def _range(name: str, **overrides: Any) -> RangeVar:
    return RangeVar(**{"name": name, "lower": -5.0, "upper": 5.0, **overrides})


def _tool(name: str, inputs: list[str], outputs: list[str]) -> ToolSpec:
    return ToolSpec(name=name, inputs=inputs, outputs=outputs)


def _paraboloid(**x_overrides: Any) -> StudySchema:
    return StudySchema(
        variables=[_range("x", **x_overrides), _range("y"), StateVar(name="f")],
        tools=[_tool("paraboloid", ["x", "y"], ["f"])],
    )


def _sellar(
    y1_guess: float | list[float] | None = None,
    y2_guess: float | list[float] | None = None,
) -> StudySchema:
    return StudySchema(
        variables=[
            _range("x1"),
            _range("z"),
            StateVar(name="y1", initial_guess=y1_guess),
            StateVar(name="y2", initial_guess=y2_guess),
            StateVar(name="f"),
        ],
        tools=[
            _tool("disc1", ["z", "x1", "y2"], ["y1"]),
            _tool("disc2", ["z", "y1"], ["y2"]),
            _tool("objective", ["x1", "z", "y1", "y2"], ["f"]),
        ],
    )


def _minimize(name: str = "f") -> list[ObjectiveSpec]:
    return [ObjectiveSpec(name=name)]


def _codes(findings: Sequence[Finding]) -> list[str]:
    return [finding.code for finding in findings]


def _only(findings: Sequence[Finding], code: str) -> Finding:
    matching = [finding for finding in findings if finding.code == code]
    assert len(matching) == 1, f"expected one {code}, got {_codes(findings)}"
    return matching[0]


def _never_called(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("validation must never call a tool")


def _paraboloid_fn(x: float, y: float) -> None:
    _never_called()


def _x_only_fn(x: float) -> None:
    _never_called()


def _disc1_fn(z: float, x1: float, y2: float) -> None:
    _never_called()


def _disc2_fn(z: float, y1: float) -> None:
    _never_called()


def _objective_fn(x1: float, z: float, y1: float, y2: float) -> None:
    _never_called()


SELLAR_REGISTRY: dict[str, Callable[..., Any]] = {
    "disc1": _disc1_fn,
    "disc2": _disc2_fn,
    "objective": _objective_fn,
}


def _validate(schema: StudySchema, **kwargs: Any) -> ValidationReport:
    kwargs.setdefault("objectives", _minimize())
    return validate_study(schema, **kwargs)


# --- parse_parameter_constraints --------------------------------------------

DESIGN_VARIABLES = [
    _range("x"),
    _range("y"),
    ChoiceVar(name="mode", choices=["a", "b"]),
]


@pytest.mark.parametrize(
    ("expression", "coefficients", "bound"),
    [
        ("x + y <= 2", {"x": 1.0, "y": 1.0}, 2.0),
        ("2*x - 3.5*y >= 1", {"x": -2.0, "y": 3.5}, -1.0),
        ("x <= y", {"x": 1.0, "y": -1.0}, 0.0),
        ("-x + 1 >= y", {"x": 1.0, "y": 1.0}, 1.0),
        ("2*x-3.5*y>=1", {"x": -2.0, "y": 3.5}, -1.0),
        ("  x   +y<=   2 ", {"x": 1.0, "y": 1.0}, 2.0),
        ("1e-3*x + y <= 1", {"x": 0.001, "y": 1.0}, 1.0),
        ("2.5 * x <= 10", {"x": 2.5}, 10.0),
        ("x + x <= 2", {"x": 2.0}, 2.0),
        ("x - x + y <= 1", {"y": 1.0}, 1.0),
        ("x + 1 <= 3 - y", {"x": 1.0, "y": 1.0}, 2.0),
        ("4 >= x", {"x": 1.0}, 4.0),
        ("+x <= 1", {"x": 1.0}, 1.0),
    ],
)
def test_parse_normalises_to_sum_of_terms_below_bound(
    expression: str, coefficients: dict[str, float], bound: float
) -> None:
    [parsed] = parse_parameter_constraints([expression], DESIGN_VARIABLES)

    assert isinstance(parsed, LinearConstraint)
    assert parsed.expression == expression
    assert parsed.coefficients == pytest.approx(coefficients)
    assert list(parsed.coefficients) == list(coefficients)
    assert parsed.bound == pytest.approx(bound)


def test_parse_handles_several_expressions_in_order() -> None:
    parsed = parse_parameter_constraints(["x <= 1", "y >= 2"], DESIGN_VARIABLES)

    assert [item.expression for item in parsed] == ["x <= 1", "y >= 2"]


def test_parse_of_no_expression_is_empty() -> None:
    assert parse_parameter_constraints([], DESIGN_VARIABLES) == []


@pytest.mark.parametrize(
    "expression",
    [
        "x + y",
        "",
        "x <= y <= 1",
        "x < 1",
        "x == 1",
        "x + <= 1",
        "<= 1",
        "x <=",
        "2x <= 1",
        "x * y <= 1",
        "x*2 <= 1",
        "x / 2 <= 1",
        "x + -y <= 1",
        "(x + y) <= 1",
        "x**2 <= 1",
    ],
)
def test_parse_rejects_unparsable_expressions(expression: str) -> None:
    with pytest.raises(ParameterConstraintError, match=re.escape(repr(expression))):
        parse_parameter_constraints([expression], DESIGN_VARIABLES)


def test_parse_rejects_a_name_that_is_not_a_design_variable() -> None:
    with pytest.raises(ParameterConstraintError, match=r"'x \+ z <= 1'.*'z'"):
        parse_parameter_constraints(["x + z <= 1"], DESIGN_VARIABLES)


def test_parse_rejects_choice_variables() -> None:
    with pytest.raises(ParameterConstraintError, match=r"'mode'.*range"):
        parse_parameter_constraints(["x + mode <= 1"], DESIGN_VARIABLES)


@pytest.mark.parametrize("expression", ["1 <= 2", "x - x <= 1", "x <= x + 1"])
def test_parse_rejects_expressions_without_a_variable(expression: str) -> None:
    with pytest.raises(ParameterConstraintError, match="no variable"):
        parse_parameter_constraints([expression], DESIGN_VARIABLES)


def test_parse_rejects_non_finite_numbers() -> None:
    with pytest.raises(ParameterConstraintError, match="finite"):
        parse_parameter_constraints(["x <= 1e999"], DESIGN_VARIABLES)


def test_parse_error_is_a_value_error() -> None:
    assert issubclass(ParameterConstraintError, ValueError)


# --- check_tool_signature ---------------------------------------------------

TOOL_XY = _tool("t", ["x", "y"], ["f"])


def test_signature_matching_the_inputs_is_clean() -> None:
    def func(x: float, y: float) -> float:
        return x + y

    assert check_tool_signature(TOOL_XY, func) == ([], [])


def test_signature_with_keyword_only_parameters_is_clean() -> None:
    def func(*, x: float, y: float) -> float:
        return x + y

    assert check_tool_signature(TOOL_XY, func) == ([], [])


def test_signature_with_missing_argument_is_an_error() -> None:
    def func(x: float, y: float, z: float) -> float:
        return x + y + z

    errors, warnings = check_tool_signature(TOOL_XY, func)

    assert warnings == []
    error = _only(errors, "SIGNATURE_MISMATCH")
    assert error.names == ("t", "x", "y")
    assert "'t'" in error.message
    assert "['x', 'y']" in error.message
    assert "'z'" in error.message


def test_signature_with_extra_graph_input_is_an_error() -> None:
    def func(x: float) -> float:
        return x

    errors, _ = check_tool_signature(TOOL_XY, func)

    error = _only(errors, "SIGNATURE_MISMATCH")
    assert "'y'" in error.message


def test_signature_with_positional_only_parameter_is_an_error() -> None:
    def func(x: float, /, y: float) -> float:
        return x + y

    errors, _ = check_tool_signature(TOOL_XY, func)

    assert _codes(errors) == ["SIGNATURE_MISMATCH"]


def test_signature_with_var_keyword_cannot_be_checked() -> None:
    def func(**kwargs: float) -> float:
        return sum(kwargs.values())

    errors, warnings = check_tool_signature(TOOL_XY, func)

    assert errors == []
    warning = _only(warnings, "SIGNATURE_UNCHECKED")
    assert warning.names == ("t",)
    assert "accepts **kwargs" in warning.message


def test_signature_that_cannot_be_introspected_is_unchecked() -> None:
    class Opaque:
        __signature__ = "not a signature"

        def __call__(self, **_kwargs: float) -> None:
            return None

    errors, warnings = check_tool_signature(TOOL_XY, Opaque())

    assert errors == []
    warning = _only(warnings, "SIGNATURE_UNCHECKED")
    assert warning.names == ("t",)
    assert "not introspectable" in warning.message


def test_defaulted_parameter_that_is_not_a_graph_input_is_a_warning() -> None:
    def func(x: float, y: float, scale: float = 2.0) -> float:
        return scale * (x + y)

    errors, warnings = check_tool_signature(TOOL_XY, func)

    assert errors == []
    warning = _only(warnings, "DEFAULTED_ARG_UNWIRED")
    assert warning.names == ("t", "scale")
    assert "'scale'" in warning.message


def test_defaulted_parameter_wired_to_a_graph_input_is_clean() -> None:
    def func(x: float, y: float = 1.0) -> float:
        return x + y

    assert check_tool_signature(TOOL_XY, func) == ([], [])


def test_var_positional_parameter_is_ignored() -> None:
    def func(x: float, y: float, *args: float) -> float:
        return x + y + sum(args)

    assert check_tool_signature(TOOL_XY, func) == ([], [])


def test_mismatch_and_unwired_default_are_both_reported() -> None:
    def func(x: float, scale: float = 2.0) -> float:
        return x * scale

    errors, warnings = check_tool_signature(TOOL_XY, func)

    assert _codes(errors) == ["SIGNATURE_MISMATCH"]
    assert _codes(warnings) == ["DEFAULTED_ARG_UNWIRED"]


# --- validate_study: all good, never calls tools ----------------------------


def test_valid_study_has_no_finding() -> None:
    schema = StudySchema(
        variables=[
            _range("x", initial=1.0),
            _range("y", initial=1.0),
            StateVar(name="f"),
        ],
        tools=[_tool("paraboloid", ["x", "y"], ["f"])],
    )

    report = validate_study(
        schema,
        objectives=_minimize(),
        constraints=[ConstraintSpec(name="f", bound=3.0)],
        parameter_constraints=["x + y <= 3"],
        registry={"paraboloid": _paraboloid_fn},
    )

    assert report.valid
    assert report.errors == ()
    assert report.warnings == ()


def test_validation_never_calls_a_tool() -> None:
    report = _validate(
        _sellar(y1_guess=1.0, y2_guess=1.0),
        constraints=[ConstraintSpec(name="y1", bound=10.0)],
        parameter_constraints=["x1 + z <= 3"],
        registry=SELLAR_REGISTRY,
    )

    assert report.valid


def test_report_is_returned_not_raised() -> None:
    report = _validate(_paraboloid(), objectives=_minimize("missing"))

    assert isinstance(report, ValidationReport)
    assert not report.valid


# --- validate_study: error codes --------------------------------------------


def test_unknown_output_is_an_error() -> None:
    report = _validate(_paraboloid(), objectives=_minimize("nope"))

    error = _only(report.errors, "UNKNOWN_OUTPUT")
    assert error.names == ("nope",)
    assert "NO_DESIGN_VARIABLES" not in _codes(report.errors)


def test_constraint_targets_are_checked_after_objectives() -> None:
    report = _validate(
        _paraboloid(),
        objectives=_minimize("first"),
        constraints=[ConstraintSpec(name="second", bound=0.0)],
    )

    unknown = [e.names for e in report.errors if e.code == "UNKNOWN_OUTPUT"]
    assert unknown == [("first",), ("second",)]


def test_declared_target_without_producer_is_an_error() -> None:
    report = _validate(_paraboloid(), objectives=_minimize("x"))

    error = _only(report.errors, "NOT_PRODUCED")
    assert error.names == ("x",)
    assert "NO_DESIGN_VARIABLES" not in _codes(report.errors)


def test_required_state_without_producer_is_an_error() -> None:
    schema = StudySchema(
        variables=[_range("x"), StateVar(name="s"), StateVar(name="f")],
        tools=[_tool("t", ["x", "s"], ["f"])],
    )

    report = _validate(schema)

    error = _only(report.errors, "UNPRODUCED_STATE")
    assert error.names == ("s",)


def test_study_without_design_variable_is_an_error() -> None:
    schema = StudySchema(
        variables=[FixedParam(name="rho", value=1.0), StateVar(name="f")],
        tools=[_tool("t", ["rho"], ["f"])],
    )

    report = _validate(schema)

    error = _only(report.errors, "NO_DESIGN_VARIABLES")
    assert error.names == ()


def test_no_design_variable_check_needs_a_target() -> None:
    report = validate_study(_paraboloid(), objectives=[])

    assert "NO_DESIGN_VARIABLES" not in _codes(report.errors)
    assert report.valid


def test_missing_registry_entry_is_an_error_for_every_schema_tool() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            StateVar(name="f"),
            StateVar(name="g"),
            StateVar(name="h"),
        ],
        tools=[
            _tool("needed", ["x"], ["f"]),
            _tool("extra", ["x"], ["g"]),
            _tool("registered", ["x"], ["h"]),
        ],
    )

    report = _validate(schema, registry={"registered": _x_only_fn})

    unregistered = [e.names for e in report.errors if e.code == "UNREGISTERED_TOOL"]
    assert unregistered == [("needed",), ("extra",)]


def test_registry_is_not_required() -> None:
    report = _validate(_paraboloid())

    assert "UNREGISTERED_TOOL" not in _codes(report.errors)


def test_signature_mismatch_is_an_error() -> None:
    def wrong(x: float) -> None:
        return None

    report = _validate(_paraboloid(), registry={"paraboloid": wrong})

    error = _only(report.errors, "SIGNATURE_MISMATCH")
    assert error.names == ("paraboloid", "x", "y")


def test_unparsable_parameter_constraints_give_one_finding_each() -> None:
    report = _validate(
        _paraboloid(),
        parameter_constraints=["x + y", "z <= 1", "x + y <= 3"],
    )

    invalid = [e for e in report.errors if e.code == "PARAMETER_CONSTRAINT_INVALID"]
    assert [e.names for e in invalid] == [("x + y",), ("z <= 1",)]
    assert "'x + y'" in invalid[0].message


def test_parameter_constraint_on_a_choice_variable_is_an_error() -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            ChoiceVar(name="mode", choices=[1, 2]),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "mode"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + mode <= 2"])

    error = _only(report.errors, "PARAMETER_CONSTRAINT_INVALID")
    assert error.names == ("x + mode <= 2",)


def test_ax_rejection_of_the_constraints_is_reported() -> None:
    schema = StudySchema(
        variables=[_range("E"), _range("y"), StateVar(name="f")],
        tools=[_tool("t", ["E", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["E + y <= 1"])

    error = _only(report.errors, "PARAMETER_CONSTRAINT_INVALID")
    assert error.names == ("E + y <= 1",)
    assert "sympy" in error.message


@pytest.mark.parametrize(
    "choices",
    [[1, 2, 3], [0.5, 1.5], ["a", "b"], [True, False]],
    ids=["int", "float", "str", "bool"],
)
def test_ax_dry_build_accepts_every_choice_type(choices: list[Any]) -> None:
    schema = StudySchema(
        variables=[
            _range("x"),
            _range("y"),
            ChoiceVar(name="mode", choices=choices),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y", "mode"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + y <= 3"])

    assert report.valid


def test_integer_range_variables_pass_the_dry_builds() -> None:
    schema = StudySchema(
        variables=[
            RangeVar(name="n", lower=0, upper=10, value_type="int", initial=2),
            _range("y"),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["n", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["n + y <= 3"])

    assert report.valid


def test_ax_dry_build_only_runs_with_parameter_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "_ax_constraint_errors", _never_called)

    report = _validate(_paraboloid())

    assert report.valid


def test_initial_values_violating_a_constraint_are_an_error() -> None:
    schema = StudySchema(
        variables=[
            _range("x", initial=4.0),
            _range("y", initial=4.0),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + y <= 2"])

    error = _only(report.errors, "INITIAL_OUT_OF_SPACE")
    assert error.names == ("x + y <= 2", "x", "y")


def test_initial_values_violating_a_lower_bound_constraint_are_an_error() -> None:
    schema = StudySchema(
        variables=[
            _range("x", initial=1.0),
            _range("y", initial=1.0),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + y >= 5"])

    assert _codes(report.errors) == ["INITIAL_OUT_OF_SPACE"]


def test_initial_values_on_the_constraint_boundary_are_accepted() -> None:
    schema = StudySchema(
        variables=[
            _range("x", initial=0.1),
            _range("y", initial=0.2),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + y <= 0.3"])

    assert report.valid


def test_initial_check_needs_every_constraint_variable_to_have_an_initial() -> None:
    schema = StudySchema(
        variables=[_range("x", initial=4.0), _range("y"), StateVar(name="f")],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["x + y <= 2"])

    assert "INITIAL_OUT_OF_SPACE" not in _codes(report.errors)


def test_initial_check_survives_a_bad_sibling_expression() -> None:
    schema = StudySchema(
        variables=[
            _range("x", initial=4.0),
            _range("y", initial=4.0),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema, parameter_constraints=["bogus", "x + y <= 2"])

    assert _codes(report.errors) == [
        "PARAMETER_CONSTRAINT_INVALID",
        "INITIAL_OUT_OF_SPACE",
    ]


def test_design_space_rejection_is_an_error(
    monkeypatch: pytest.MonkeyPatch, design_space_class: Any
) -> None:
    original = design_space_class.add_variable

    def reject_y(self: Any, name: str, **kwargs: Any) -> None:
        if name == "y":
            raise ValueError("rejected by gemseo")
        original(self, name, **kwargs)

    monkeypatch.setattr(design_space_class, "add_variable", reject_y)

    report = _validate(_paraboloid())

    error = _only(report.errors, "DESIGN_SPACE_INVALID")
    assert error.names == ("y",)
    assert "rejected by gemseo" in error.message


def test_design_space_construction_failure_is_an_error(
    monkeypatch: pytest.MonkeyPatch, design_space_class: Any
) -> None:
    def fail_construction(self: Any, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("cannot construct")

    monkeypatch.setattr(design_space_class, "__init__", fail_construction)

    report = _validate(_paraboloid())

    error = _only(report.errors, "DESIGN_SPACE_INVALID")
    assert error.names == ("x", "y")
    assert "cannot construct" in error.message


def test_every_applicable_error_is_reported_in_order() -> None:
    report = validate_study(
        _paraboloid(),
        objectives=_minimize("nope"),
        constraints=[ConstraintSpec(name="x", bound=0.0)],
        parameter_constraints=["x + y <= 1"],
        registry={},
    )

    assert _codes(report.errors) == [
        "UNKNOWN_OUTPUT",
        "NOT_PRODUCED",
        "UNREGISTERED_TOOL",
        "PARAMETER_CONSTRAINT_INVALID",
    ]


# --- validate_study: warning codes ------------------------------------------


def test_coupling_without_initial_guess_is_a_warning() -> None:
    report = _validate(_sellar(y1_guess=None, y2_guess=1.0))

    warning = _only(report.warnings, "COUPLING_DEFAULT_GUESS")
    assert warning.names == ("y1",)
    assert "0.0" in warning.message
    assert report.valid


def test_coupling_with_initial_guesses_is_clean() -> None:
    report = _validate(_sellar(y1_guess=1.0, y2_guess=[1.0]))

    assert "COUPLING_DEFAULT_GUESS" not in _codes(report.warnings)


def test_variable_used_by_no_tool_is_a_warning() -> None:
    schema = StudySchema(
        variables=[_range("x"), _range("spare"), StateVar(name="f")],
        tools=[_tool("t", ["x"], ["f"])],
    )

    report = _validate(schema)

    warning = _only(report.warnings, "UNUSED_VARIABLE")
    assert warning.names == ("spare",)


def test_variable_only_produced_by_a_tool_is_used() -> None:
    report = _validate(_paraboloid())

    assert "UNUSED_VARIABLE" not in _codes(report.warnings)


def test_tool_not_needed_by_the_targets_is_a_warning() -> None:
    schema = StudySchema(
        variables=[_range("x"), StateVar(name="f"), StateVar(name="g")],
        tools=[_tool("needed", ["x"], ["f"]), _tool("extra", ["x"], ["g"])],
    )

    report = _validate(schema)

    warning = _only(report.warnings, "UNUSED_TOOL")
    assert warning.names == ("extra",)


def test_unused_tool_is_not_reported_when_targets_do_not_resolve() -> None:
    schema = StudySchema(
        variables=[_range("x"), StateVar(name="f"), StateVar(name="g")],
        tools=[_tool("needed", ["x"], ["f"]), _tool("extra", ["x"], ["g"])],
    )

    report = _validate(schema, objectives=_minimize("nope"))

    assert "UNUSED_TOOL" not in _codes(report.warnings)


def test_signature_warnings_are_reported_for_registered_tools() -> None:
    def kwargs_tool(**kwargs: float) -> None:
        return None

    def defaulted_tool(x: float, y: float, scale: float = 1.0) -> None:
        return None

    schema = StudySchema(
        variables=[_range("x"), _range("y"), StateVar(name="f"), StateVar(name="g")],
        tools=[_tool("loose", ["x"], ["f"]), _tool("tuned", ["x", "y"], ["g"])],
    )
    registry = {"loose": kwargs_tool, "tuned": defaulted_tool}

    report = _validate(
        schema,
        objectives=_minimize("f"),
        constraints=[ConstraintSpec(name="g", bound=1.0)],
        registry=registry,
    )

    assert report.valid
    assert _only(report.warnings, "SIGNATURE_UNCHECKED").names == ("loose",)
    assert _only(report.warnings, "DEFAULTED_ARG_UNWIRED").names == ("tuned", "scale")


def test_some_but_not_all_initial_values_is_a_warning() -> None:
    report = _validate(_paraboloid(initial=1.0))

    warning = _only(report.warnings, "PARTIAL_INITIAL")
    assert warning.names == ("y",)
    assert "x0" in warning.message


@pytest.mark.parametrize("with_initial", [True, False])
def test_all_or_no_initial_values_is_clean(with_initial: bool) -> None:
    overrides = {"initial": 1.0} if with_initial else {}
    schema = StudySchema(
        variables=[
            _range("x", **overrides),
            _range("y", **overrides),
            StateVar(name="f"),
        ],
        tools=[_tool("t", ["x", "y"], ["f"])],
    )

    report = _validate(schema)

    assert "PARTIAL_INITIAL" not in _codes(report.warnings)
