"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import hashlib
import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from mdo_framework.schema import (
    MAX_NAME_LENGTH,
    ChoiceVar,
    ConstraintSpec,
    Finding,
    FixedParam,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolNode,
    ToolSpec,
    ValidationReport,
    Variable,
    report_from_validation_error,
)


def _range(name: str = "x", **overrides: Any) -> RangeVar:
    return RangeVar(**{"name": name, "lower": 0.0, "upper": 1.0, **overrides})


def _paraboloid() -> StudySchema:
    return StudySchema(
        variables=[
            RangeVar(name="x", lower=-5, upper=5, initial=1),
            RangeVar(name="y", lower=-5, upper=5),
            StateVar(name="f"),
        ],
        tools=[ToolSpec(name="paraboloid", inputs=["x", "y"], outputs=["f"])],
    )


def _messages(exc_info: pytest.ExceptionInfo[ValidationError]) -> str:
    return " | ".join(error["msg"] for error in exc_info.value.errors())


def _finding_codes(exc_info: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    report = report_from_validation_error(exc_info.value)
    return [finding.code for finding in report.errors]


# --- Name -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["x", "_x", "Mach_2", "a" * MAX_NAME_LENGTH])
def test_name_accepts_valid_identifiers(name: str) -> None:
    assert FixedParam(name=name, value=1).name == name


@pytest.mark.parametrize("name", ["1x", "a-b", "a b", "", "é"])
def test_name_rejects_bad_pattern(name: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FixedParam(name=name, value=1)
    assert exc_info.value.errors()[0]["type"] == "string_pattern_mismatch"


def test_name_rejects_too_long() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FixedParam(name="a" * (MAX_NAME_LENGTH + 1), value=1)
    assert exc_info.value.errors()[0]["type"] == "string_too_long"


def test_name_rejects_hard_keyword() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FixedParam(name="class", value=1)
    assert "'class' is a Python keyword and cannot be a tool argument" in _messages(
        exc_info
    )


@pytest.mark.parametrize("name", ["type", "match", "case", "_"])
def test_name_accepts_soft_keywords(name: str) -> None:
    assert FixedParam(name=name, value=1).name == name


# --- Finite -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), True, "1.5", None]
)
def test_finite_rejects_non_finite_bool_and_str(value: Any) -> None:
    with pytest.raises(ValidationError):
        RangeVar(name="x", lower=value, upper=2.0)


def test_finite_accepts_int_as_float() -> None:
    variable = RangeVar(name="x", lower=0, upper=2)
    assert variable.lower == 0.0
    assert isinstance(variable.lower, float)
    assert isinstance(variable.upper, float)


def test_finite_accepts_json_integer() -> None:
    variable = RangeVar.model_validate_json('{"name": "x", "lower": 0, "upper": 2}')
    assert isinstance(variable.lower, float)


# --- RangeVar ---------------------------------------------------------------


def test_range_defaults() -> None:
    variable = _range()
    assert variable.kind == "range"
    assert variable.value_type == "float"
    assert variable.initial is None
    assert variable.units is None


@pytest.mark.parametrize(("lower", "upper"), [(1.0, 1.0), (2.0, 1.0)])
def test_range_rejects_lower_not_below_upper(lower: float, upper: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RangeVar(name="x", lower=lower, upper=upper)
    assert "lower" in _messages(exc_info)
    assert "upper" in _messages(exc_info)


def test_range_int_accepts_integral_values() -> None:
    variable = RangeVar(name="n", lower=1, upper=4, value_type="int", initial=2)
    assert variable.value_type == "int"
    assert variable.initial == 2.0


def test_range_int_rejects_non_integral_bound() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RangeVar(name="n", lower=0.5, upper=4, value_type="int")
    assert "integral" in _messages(exc_info)


def test_range_int_rejects_non_integral_initial() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RangeVar(name="n", lower=0, upper=4, value_type="int", initial=1.5)
    assert "integral" in _messages(exc_info)


@pytest.mark.parametrize("initial", [-0.1, 1.1])
def test_range_rejects_initial_out_of_bounds(initial: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _range(initial=initial)
    assert "initial" in _messages(exc_info)


@pytest.mark.parametrize("initial", [0.0, 0.5, 1.0])
def test_range_accepts_initial_within_bounds(initial: float) -> None:
    assert _range(initial=initial).initial == initial


def test_range_reports_all_violations_in_one_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RangeVar(name="n", lower=3.5, upper=1.5, value_type="int", initial=9.5)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    message = errors[0]["msg"]
    assert "lower" in message and "upper" in message
    assert "integral" in message
    assert "initial" in message


def test_range_rejects_unknown_value_type() -> None:
    with pytest.raises(ValidationError):
        _range(value_type="complex")


# --- ChoiceVar --------------------------------------------------------------


def test_choice_defaults() -> None:
    variable = ChoiceVar(name="m", choices=["a", "b"])
    assert variable.kind == "choice"
    assert variable.initial is None
    assert variable.units is None


def test_choice_rejects_single_choice_and_points_to_fixed() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=["a"])
    assert "fixed" in _messages(exc_info)


def test_choice_rejects_empty_choices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=[])
    assert "fixed" in _messages(exc_info)


@pytest.mark.parametrize("choices", [["a", "a"], [1, 2, 1], [True, True]])
def test_choice_rejects_duplicate_choices(choices: list[Any]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=choices)
    assert "unique" in _messages(exc_info)


@pytest.mark.parametrize("choices", [[1, True], [1, 2.0], ["1", 1], [True, "a"]])
def test_choice_rejects_mixed_types(choices: list[Any]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=choices)
    assert "same type" in _messages(exc_info)


def test_choice_rejects_non_finite_choice() -> None:
    with pytest.raises(ValidationError):
        ChoiceVar(name="m", choices=[1.0, float("nan")])


def test_choice_rejects_initial_not_in_choices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=["a", "b"], initial="c")
    assert "initial" in _messages(exc_info)


def test_choice_rejects_bool_initial_for_int_choices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceVar(name="m", choices=[0, 1], initial=True)
    assert "initial" in _messages(exc_info)


def test_choice_accepts_initial_in_choices() -> None:
    assert ChoiceVar(name="m", choices=[0, 1], initial=1).initial == 1


@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        ([True, False], "bool"),
        ([1, 2, 3], "int"),
        ([0.5, 1.5], "float"),
        (["a", "b"], "str"),
    ],
)
def test_choice_value_type_is_derived_from_choices(
    choices: list[Any], expected: str
) -> None:
    assert ChoiceVar(name="m", choices=choices).value_type == expected


def test_choice_value_type_is_not_serialized() -> None:
    assert "value_type" not in ChoiceVar(name="m", choices=[1, 2]).model_dump()


def test_choice_value_type_is_read_only() -> None:
    variable = ChoiceVar(name="m", choices=[1, 2])
    with pytest.raises(ValidationError):
        variable.value_type = "str"  # type: ignore[misc]


# --- FixedParam -------------------------------------------------------------


def test_fixed_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        FixedParam(name="p", value=float("nan"))


@pytest.mark.parametrize("value", ["steel", 3, 2.5, True])
def test_fixed_accepts_scalars_with_their_type(value: Any) -> None:
    parameter = FixedParam(name="p", value=value)
    assert parameter.kind == "fixed"
    assert parameter.value == value
    assert type(parameter.value) is type(value)


def test_fixed_rejects_list_value() -> None:
    with pytest.raises(ValidationError):
        FixedParam(name="p", value=[1, 2])


# --- StateVar ---------------------------------------------------------------


def test_state_defaults() -> None:
    variable = StateVar(name="f")
    assert variable.kind == "state"
    assert variable.initial_guess is None


def test_state_rejects_empty_initial_guess_list() -> None:
    with pytest.raises(ValidationError):
        StateVar(name="f", initial_guess=[])


def test_state_accepts_list_and_scalar_initial_guess() -> None:
    assert StateVar(name="f", initial_guess=[1, 2.5]).initial_guess == [1.0, 2.5]
    assert StateVar(name="f", initial_guess=3).initial_guess == 3.0


def test_state_rejects_non_finite_initial_guess() -> None:
    with pytest.raises(ValidationError):
        StateVar(name="f", initial_guess=[1.0, float("inf")])


# --- Discriminator / extra --------------------------------------------------


def test_unknown_kind_is_rejected_by_discriminator() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Variable).validate_python(
            {"kind": "choise", "name": "m", "choices": ["a", "b"]}
        )
    assert exc_info.value.errors()[0]["type"] == "union_tag_invalid"


def test_missing_kind_is_rejected_in_dict() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Variable).validate_python({"name": "x", "lower": 0, "upper": 1})
    assert exc_info.value.errors()[0]["type"] == "union_tag_not_found"


def test_discriminator_selects_each_kind() -> None:
    adapter = TypeAdapter(Variable)
    assert isinstance(
        adapter.validate_python({"kind": "range", "name": "x", "lower": 0, "upper": 1}),
        RangeVar,
    )
    assert isinstance(
        adapter.validate_python({"kind": "choice", "name": "m", "choices": [1, 2]}),
        ChoiceVar,
    )
    assert isinstance(
        adapter.validate_python({"kind": "fixed", "name": "p", "value": 1}),
        FixedParam,
    )
    assert isinstance(adapter.validate_python({"kind": "state", "name": "f"}), StateVar)


def test_extra_keys_are_forbidden() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RangeVar(name="x", lower=0, upper=1, param_type="continuous")
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_models_are_frozen() -> None:
    variable = _range()
    with pytest.raises(ValidationError):
        variable.lower = -1.0  # type: ignore[misc]


# --- ToolSpec ---------------------------------------------------------------


def test_tool_defaults() -> None:
    tool = ToolSpec(name="t")
    assert tool.inputs == []
    assert tool.outputs == []
    assert tool.fidelity == "high"


def test_tool_rejects_duplicate_input() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolSpec(name="t", inputs=["x", "x"])
    assert "inputs" in _messages(exc_info)


def test_tool_rejects_duplicate_output() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolSpec(name="t", outputs=["f", "f"])
    assert "outputs" in _messages(exc_info)


def test_tool_rejects_name_both_input_and_output() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolSpec(name="t", inputs=["x"], outputs=["x"])
    assert "both" in _messages(exc_info)


def test_tool_rejects_keyword_argument_name() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="t", inputs=["lambda"])


def test_tool_rejects_bad_fidelity_name() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="t", fidelity="not valid")


def test_tool_node_defaults_and_is_the_node_part_of_a_tool_spec() -> None:
    node = ToolNode(name="t")
    assert node.fidelity == "high"
    assert node.deterministic is True
    assert node.thread_safe is False
    assert node.arg_map == {}
    assert isinstance(ToolSpec(name="t"), ToolNode)


def test_tool_options_are_inherited_by_the_tool_spec() -> None:
    tool = ToolSpec(
        name="t",
        inputs=["x", "y"],
        deterministic=False,
        thread_safe=True,
        arg_map={"x": "a", "y": "b"},
    )
    assert tool.deterministic is False
    assert tool.thread_safe is True
    assert tool.arg_map == {"x": "a", "y": "b"}
    assert tool.model_dump()["arg_map"] == {"x": "a", "y": "b"}


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_deterministic_must_be_a_boolean(value: Any) -> None:
    with pytest.raises(ValidationError):
        ToolNode(name="t", deterministic=value)


@pytest.mark.parametrize("value", ["true", 0, 1, None])
def test_thread_safe_must_be_a_boolean(value: Any) -> None:
    with pytest.raises(ValidationError):
        ToolNode(name="t", thread_safe=value)


def test_arg_map_rejects_two_graph_inputs_feeding_one_argument() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolNode(name="t", arg_map={"x": "a", "y": "a"})
    assert "unique" in _messages(exc_info)
    assert "'a'" in _messages(exc_info)


@pytest.mark.parametrize(
    "arg_map",
    [
        {"not valid": "a"},
        {"x": "not valid"},
        {"x": "lambda"},
        {"lambda": "a"},
        {"x": ""},
    ],
)
def test_arg_map_names_follow_the_name_rule(arg_map: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ToolNode(name="t", arg_map=arg_map)


def test_arg_map_key_need_not_be_a_tool_input() -> None:
    node = ToolNode(name="t", arg_map={"ghost": "a"})
    assert node.arg_map == {"ghost": "a"}


def test_arg_map_may_map_an_input_to_its_own_name() -> None:
    assert ToolNode(name="t", arg_map={"x": "x"}).arg_map == {"x": "x"}


def test_tool_options_survive_a_json_round_trip() -> None:
    study = StudySchema(
        variables=[RangeVar(name="x", lower=0, upper=1), StateVar(name="f")],
        tools=[
            ToolSpec(
                name="t",
                inputs=["x"],
                outputs=["f"],
                deterministic=False,
                thread_safe=True,
                arg_map={"x": "a"},
            )
        ],
    )
    restored = StudySchema.model_validate_json(study.model_dump_json())
    assert restored == study
    assert restored.tool("t").arg_map == {"x": "a"}
    assert restored.tool("t").deterministic is False
    assert restored.tool("t").thread_safe is True


def test_tool_options_change_the_content_hash() -> None:
    def study(**options: Any) -> StudySchema:
        return StudySchema(
            variables=[RangeVar(name="x", lower=0, upper=1), StateVar(name="f")],
            tools=[ToolSpec(name="t", inputs=["x"], outputs=["f"], **options)],
        )

    hashes = {
        study().content_hash(),
        study(deterministic=False).content_hash(),
        study(thread_safe=True).content_hash(),
        study(arg_map={"x": "a"}).content_hash(),
    }
    assert len(hashes) == 4


@pytest.mark.parametrize("extra", ["inputs", "outputs"])
def test_tool_node_rejects_edge_fields(extra: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolNode(name="t", **{extra: ["x"]})
    assert "extra" in _messages(exc_info).lower()


def test_tool_node_rejects_bad_name() -> None:
    with pytest.raises(ValidationError):
        ToolNode(name="not valid")


# --- ObjectiveSpec / ConstraintSpec -----------------------------------------


def test_objective_defaults() -> None:
    objective = ObjectiveSpec(name="f")
    assert objective.minimize is True
    assert objective.threshold is None


def test_objective_rejects_non_finite_threshold() -> None:
    with pytest.raises(ValidationError):
        ObjectiveSpec(name="f", threshold=float("inf"))


def test_constraint_defaults_and_validation() -> None:
    constraint = ConstraintSpec(name="c", bound=0)
    assert constraint.op == "<="
    assert constraint.bound == 0.0
    assert ConstraintSpec(name="c", bound=1, op=">=").op == ">="
    with pytest.raises(ValidationError):
        ConstraintSpec(name="c", bound=1, op="<")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ConstraintSpec(name="c", bound=float("nan"))


# --- StudySchema structure --------------------------------------------------


def test_empty_study_is_valid() -> None:
    study = StudySchema()
    assert study.schema_version == "1"
    assert study.variables == []
    assert study.tools == []


def test_valid_paraboloid_study() -> None:
    study = _paraboloid()
    assert [variable.name for variable in study.variables] == ["x", "y", "f"]


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StudySchema.model_validate({"schema_version": "2"})


def test_duplicate_variable_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(variables=[_range("x"), StateVar(name="x")])
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["DUPLICATE_NAME"]
    assert report.errors[0].names == ("x",)


def test_duplicate_tool_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(tools=[ToolSpec(name="t"), ToolSpec(name="t")])
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["DUPLICATE_NAME"]
    assert report.errors[0].names == ("t",)


def test_tool_may_share_a_variable_name() -> None:
    study = StudySchema(
        variables=[_range("x"), StateVar(name="f")],
        tools=[ToolSpec(name="f", inputs=["x"], outputs=["f"])],
    )
    assert study.tool("f").outputs == ["f"]
    assert study.variable("f").kind == "state"


@pytest.mark.parametrize(
    "tool",
    [
        ToolSpec(name="t", inputs=["ghost"]),
        ToolSpec(name="t", outputs=["ghost"]),
    ],
)
def test_undeclared_reference(tool: ToolSpec) -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(variables=[_range("x")], tools=[tool])
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["UNDECLARED_REF"]
    assert report.errors[0].names == ("ghost", "t")


def test_duplicate_producer() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(
            variables=[_range("x"), StateVar(name="f")],
            tools=[
                ToolSpec(name="low", inputs=["x"], outputs=["f"]),
                ToolSpec(name="high", inputs=["x"], outputs=["f"]),
            ],
        )
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["DUPLICATE_PRODUCER"]
    assert report.errors[0].names == ("f", "low", "high")


@pytest.mark.parametrize(
    "design_variable",
    [_range("x"), ChoiceVar(name="x", choices=["a", "b"])],
)
def test_produced_design_variable(design_variable: RangeVar | ChoiceVar) -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(
            variables=[design_variable],
            tools=[ToolSpec(name="t", outputs=["x"])],
        )
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["PRODUCED_DESIGN_VAR"]
    assert report.errors[0].names == ("x", "t")


def test_produced_fixed_parameter() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(
            variables=[FixedParam(name="p", value=1)],
            tools=[ToolSpec(name="t", outputs=["p"])],
        )
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["PRODUCED_FIXED"]
    assert report.errors[0].names == ("p", "t")


def test_unconnected_tool_is_allowed() -> None:
    study = StudySchema(tools=[ToolSpec(name="t")])
    assert study.producers() == {}


def test_all_structural_violations_are_reported_in_one_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(
            variables=[
                _range("x"),
                _range("x", lower=2.0, upper=3.0),
                FixedParam(name="p", value=1),
                StateVar(name="f"),
            ],
            tools=[
                ToolSpec(name="a", inputs=["ghost"], outputs=["f", "x"]),
                ToolSpec(name="b", outputs=["f", "p"]),
            ],
        )
    assert len(exc_info.value.errors()) == 1
    assert exc_info.value.errors()[0]["type"] == "study_structure"
    assert _finding_codes(exc_info) == [
        "DUPLICATE_NAME",
        "UNDECLARED_REF",
        "DUPLICATE_PRODUCER",
        "PRODUCED_DESIGN_VAR",
        "PRODUCED_FIXED",
    ]


def test_study_error_message_lists_the_findings() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema(variables=[StateVar(name="f"), StateVar(name="f")])
    assert "DUPLICATE_NAME" in str(exc_info.value)


# --- report_from_validation_error -------------------------------------------


def test_report_from_field_error_is_schema_invalid() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema.model_validate(
            {"variables": [{"kind": "range", "name": "1x", "lower": 0, "upper": 1}]}
        )
    report = report_from_validation_error(exc_info.value)
    assert not report.valid
    assert len(report.errors) == 1
    finding = report.errors[0]
    assert finding.code == "SCHEMA_INVALID"
    assert finding.message.startswith("variables.0.range.name: ")
    assert finding.names == ("variables", "range", "name")


def test_report_from_error_without_location_is_schema_invalid() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Variable).validate_python(
            {"kind": "range", "name": "x", "lower": 1, "upper": 0}
        )
    finding = report_from_validation_error(exc_info.value).errors[0]
    assert finding.code == "SCHEMA_INVALID"
    assert finding.names == ("range",)


def test_report_from_schema_version_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StudySchema.model_validate({"schema_version": "9"})
    report = report_from_validation_error(exc_info.value)
    assert [finding.code for finding in report.errors] == ["SCHEMA_INVALID"]
    assert report.errors[0].names == ("schema_version",)


# --- Lookups, producers, hash -----------------------------------------------


def test_producers_maps_variable_to_tool() -> None:
    study = StudySchema(
        variables=[_range("x"), StateVar(name="a"), StateVar(name="b")],
        tools=[
            ToolSpec(name="t1", inputs=["x"], outputs=["a"]),
            ToolSpec(name="t2", inputs=["a"], outputs=["b"]),
        ],
    )
    assert study.producers() == {"a": "t1", "b": "t2"}


def test_variable_lookup() -> None:
    study = _paraboloid()
    assert study.variable("f") == StateVar(name="f")
    with pytest.raises(KeyError) as exc_info:
        study.variable("missing")
    assert exc_info.value.args == ("missing",)


def test_tool_lookup() -> None:
    study = _paraboloid()
    assert study.tool("paraboloid").outputs == ["f"]
    with pytest.raises(KeyError) as exc_info:
        study.tool("missing")
    assert exc_info.value.args == ("missing",)


def test_content_hash_matches_canonical_json_digest() -> None:
    study = _paraboloid()
    canonical = json.dumps(
        study.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    assert study.content_hash() == hashlib.sha256(canonical.encode()).hexdigest()


def test_content_hash_is_stable_for_equal_content() -> None:
    assert _paraboloid().content_hash() == _paraboloid().content_hash()


def test_content_hash_changes_when_content_changes() -> None:
    other = StudySchema(
        variables=[
            RangeVar(name="x", lower=-5, upper=6, initial=1),
            RangeVar(name="y", lower=-5, upper=5),
            StateVar(name="f"),
        ],
        tools=[ToolSpec(name="paraboloid", inputs=["x", "y"], outputs=["f"])],
    )
    assert other.content_hash() != _paraboloid().content_hash()


def test_content_hash_changes_when_variables_are_reordered() -> None:
    study = _paraboloid()
    x, y, f = study.variables
    reordered = StudySchema(variables=[y, x, f], tools=study.tools)
    assert reordered.content_hash() != study.content_hash()


# --- Report / error types ---------------------------------------------------


def test_validation_report_valid_is_a_computed_field() -> None:
    assert ValidationReport().valid is True
    assert ValidationReport().model_dump()["valid"] is True
    failing = ValidationReport(errors=(Finding(code="X", message="boom"),))
    assert failing.valid is False
    assert failing.model_dump()["valid"] is False
    assert json.loads(failing.model_dump_json())["valid"] is False


def test_warnings_do_not_invalidate_a_report() -> None:
    report = ValidationReport(warnings=(Finding(code="W", message="careful"),))
    assert report.valid is True


def test_finding_defaults() -> None:
    assert Finding(code="X", message="m").names == ()


def test_study_validation_error_str_lists_every_error() -> None:
    report = ValidationReport(
        errors=(
            Finding(code="A_CODE", message="first", names=("x",)),
            Finding(code="B_CODE", message="second"),
        ),
        warnings=(Finding(code="W_CODE", message="ignored"),),
    )
    error = StudyValidationError(report)
    assert isinstance(error, ValueError)
    assert error.report is report
    assert str(error) == "A_CODE: first; B_CODE: second"


# --- JSON round trip --------------------------------------------------------


def test_json_round_trip() -> None:
    study = StudySchema(
        variables=[
            RangeVar(name="x", lower=-5, upper=5, initial=1, units="m"),
            RangeVar(name="n", lower=1, upper=8, value_type="int", initial=2),
            ChoiceVar(name="mat", choices=["al", "steel"], initial="al"),
            ChoiceVar(name="flag", choices=[True, False]),
            ChoiceVar(name="k", choices=[1, 2, 3]),
            ChoiceVar(name="r", choices=[0.5, 1.5]),
            FixedParam(name="rho", value=2.7),
            FixedParam(name="label", value="a"),
            StateVar(name="f", initial_guess=[1.0, 2.0]),
            StateVar(name="g", initial_guess=0.0),
        ],
        tools=[
            ToolSpec(name="t", inputs=["x", "n", "mat"], outputs=["f"], fidelity="low")
        ],
    )
    restored = StudySchema.model_validate_json(study.model_dump_json())
    assert restored == study
    assert restored.content_hash() == study.content_hash()
    assert type(restored.variable("k").choices[0]) is int
    assert type(restored.variable("flag").choices[0]) is bool
