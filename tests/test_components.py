"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from gemseo.core.grammars.errors import InvalidDataError

from mdo_framework.core.components import ToolComponent
from mdo_framework.core.errors import (
    EvaluationError,
    InfeasiblePointError,
    ToolError,
    ToolExecutionError,
    ToolOutputError,
)


def _array(value: float) -> np.ndarray:
    return np.array([value])


def _add(x: float, y: float) -> float:
    return x + y


def _two_outputs(a: float) -> dict[str, float]:
    return {"b": a * 2, "c": a + 1}


def _component(
    func: Callable[..., Any],
    outputs: list[str],
    inputs: list[str] | None = None,
    **options: Any,
) -> ToolComponent:
    return ToolComponent("tool", func, inputs or ["a"], outputs, **options)


def _run_a(comp: ToolComponent, a: float = 1.0) -> dict[str, np.ndarray]:
    return comp.execute({"a": _array(a)})


# --- Execution --------------------------------------------------------------


def test_a_bare_value_feeds_the_single_output() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"])

    out = comp.execute({"x": _array(2.0), "y": _array(3.0)})

    np.testing.assert_allclose(out["z"], [5.0])


def test_a_dict_feeds_the_single_output() -> None:
    comp = _component(lambda a: {"f": a * 3}, ["f"])

    np.testing.assert_allclose(_run_a(comp, 2.0)["f"], [6.0])


def test_a_dict_feeds_several_outputs_by_name() -> None:
    out = _run_a(_component(_two_outputs, ["b", "c"]), 10.0)

    np.testing.assert_allclose(out["b"], [20.0])
    np.testing.assert_allclose(out["c"], [11.0])


def test_the_order_of_the_dict_and_of_the_declared_outputs_never_matters() -> None:
    def reversed_keys(a: float) -> dict[str, float]:
        return {"c": a + 1, "b": a * 2}

    out = _run_a(_component(reversed_keys, ["b", "c"]), 10.0)
    swapped = _run_a(_component(reversed_keys, ["c", "b"]), 10.0)

    for result in (out, swapped):
        np.testing.assert_allclose(result["b"], [20.0])
        np.testing.assert_allclose(result["c"], [11.0])


def test_outputs_are_one_dimensional_float_arrays() -> None:
    out = _run_a(_component(lambda a: 3, ["f"]))

    assert out["f"].dtype == np.float64
    assert out["f"].shape == (1,)


def test_a_vector_output_keeps_its_values() -> None:
    out = _run_a(_component(lambda a: [a, 2 * a, 3 * a], ["v"]), 1.0)

    np.testing.assert_allclose(out["v"], [1.0, 2.0, 3.0])


def test_an_array_is_a_vector_value_of_a_single_output() -> None:
    out = _run_a(_component(lambda a: np.array([a, 2 * a]), ["v"]), 1.0)

    np.testing.assert_allclose(out["v"], [1.0, 2.0])


def test_a_parameter_name_that_is_not_an_input_fails_naming_the_tool() -> None:
    def mismatch(p: float, q: float) -> float:
        return p + q

    comp = ToolComponent("mismatch", mismatch, ["x", "y"], ["z"])

    with pytest.raises(ToolExecutionError) as exc_info:
        comp.execute({"x": _array(1.0), "y": _array(2.0)})

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "Tool 'mismatch'" in str(exc_info.value)


def test_the_function_receives_keyword_arguments_in_any_declared_order() -> None:
    comp = ToolComponent("sub", lambda x, y: x - y, ["y", "x"], ["z"])

    out = comp.execute({"x": _array(5.0), "y": _array(2.0)})

    np.testing.assert_allclose(out["z"], [3.0])


# --- Output contract --------------------------------------------------------


@pytest.mark.parametrize("result", [(20.0, 11.0), [20.0, 11.0], 5.0])
def test_several_outputs_require_a_dict(result: Any) -> None:
    comp = _component(lambda a: result, ["b", "c"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    message = str(exc_info.value)
    assert "Tool 'tool'" in message
    assert "dict keyed by output name" in message
    assert exc_info.value.tool == "tool"


@pytest.mark.parametrize("outputs", [["f"], ["b", "c"]])
@pytest.mark.parametrize("result", [(1.0,), (1.0, 2.0), (1.0, 2.0, 3.0)])
def test_a_tuple_is_an_error_whatever_the_number_of_outputs(
    outputs: list[str], result: tuple[float, ...]
) -> None:
    comp = _component(lambda a: result, outputs)

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    message = str(exc_info.value)
    assert "Tool 'tool'" in message
    assert "tuple" in message
    assert "dict" in message
    assert "single value" in message
    assert exc_info.value.tool == "tool"


@pytest.mark.parametrize("result", [[1.0], [1.0, 2.0], np.array([1.0, 2.0])])
def test_a_list_or_array_is_a_vector_value_of_a_single_output(result: Any) -> None:
    out = _run_a(_component(lambda a: result, ["v"]))

    np.testing.assert_allclose(out["v"], np.atleast_1d(result))


def test_a_missing_key_is_reported() -> None:
    comp = _component(lambda a: {"b": a}, ["b", "c"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    assert "missing" in str(exc_info.value)
    assert "'c'" in str(exc_info.value)
    assert "Tool 'tool'" in str(exc_info.value)


def test_an_unexpected_key_is_reported() -> None:
    comp = _component(lambda a: {"b": a, "c": a, "d": a}, ["b", "c"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    assert "unexpected" in str(exc_info.value)
    assert "'d'" in str(exc_info.value)


def test_a_dict_with_the_wrong_key_for_a_single_output_is_reported() -> None:
    comp = _component(lambda a: {"g": a}, ["f"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    assert "'f'" in str(exc_info.value)
    assert "'g'" in str(exc_info.value)


def test_more_values_than_outputs_is_an_error() -> None:
    comp = _component(lambda a: (a, a, a), ["b", "c"])

    with pytest.raises(ToolOutputError):
        _run_a(comp)


@pytest.mark.parametrize("outputs", [["f"], ["b", "c"]])
def test_none_is_an_error(outputs: list[str]) -> None:
    comp = _component(lambda a: None, outputs)

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    assert "None" in str(exc_info.value)
    assert "Tool 'tool'" in str(exc_info.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_an_error_listing_the_output(bad: float) -> None:
    comp = _component(lambda a: {"b": a, "c": bad}, ["b", "c"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    message = str(exc_info.value)
    assert "non-finite" in message
    assert "'c'" in message
    assert "'b'" not in message


def test_a_non_finite_element_of_a_vector_is_an_error() -> None:
    comp = _component(lambda a: [1.0, float("nan")], ["v"])

    with pytest.raises(ToolOutputError, match="non-finite"):
        _run_a(comp)


@pytest.mark.parametrize("bad", ["text", object(), [1.0, "x"], {"k": 1}])
def test_a_non_numeric_value_is_an_error(bad: Any) -> None:
    comp = _component(lambda a: {"f": bad}, ["f"])

    with pytest.raises(ToolOutputError) as exc_info:
        _run_a(comp)

    assert "not numeric" in str(exc_info.value)
    assert "'f'" in str(exc_info.value)


def test_output_errors_are_evaluation_errors_and_value_errors() -> None:
    comp = _component(lambda a: None, ["f"])

    with pytest.raises(EvaluationError) as exc_info:
        _run_a(comp)

    assert isinstance(exc_info.value, ToolError)
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.code == "OUTPUT_INVALID"


# --- Failures inside the tool -----------------------------------------------


@pytest.mark.parametrize(
    "cause",
    [
        RuntimeError("solver diverged"),
        KeyError("table"),
        OSError("disk full"),
        TypeError("bad operand"),
        ZeroDivisionError("division by zero"),
        subprocess.CalledProcessError(2, ["solver", "--run"]),
    ],
    ids=lambda cause: type(cause).__name__,
)
def test_an_exception_of_the_tool_becomes_a_tool_execution_error(
    cause: Exception,
) -> None:
    def failing(a: float) -> float:
        raise cause

    with pytest.raises(ToolExecutionError) as exc_info:
        _run_a(_component(failing, ["f"]))

    error = exc_info.value
    assert error.__cause__ is cause
    assert error.code == "TOOL_FAILED"
    assert error.tool == "tool"
    assert str(error).startswith("Tool 'tool': ")
    assert type(cause).__name__ in str(error)


def test_the_original_message_is_kept() -> None:
    def failing(a: float) -> float:
        raise RuntimeError("solver diverged at step 7")

    with pytest.raises(ToolExecutionError, match="solver diverged at step 7"):
        _run_a(_component(failing, ["f"]))


def test_an_infeasible_point_keeps_its_class_and_gets_the_tool_name() -> None:
    def mesher(a: float) -> float:
        raise InfeasiblePointError("mesh cannot be generated")

    with pytest.raises(InfeasiblePointError) as exc_info:
        _run_a(_component(mesher, ["f"]))

    error = exc_info.value
    assert error.code == "POINT_INFEASIBLE"
    assert error.tool == "tool"
    assert str(error) == "Tool 'tool': mesh cannot be generated"


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit, MemoryError])
def test_interpreter_level_exceptions_propagate_unwrapped(
    signal: type[BaseException],
) -> None:
    def interrupted(a: float) -> float:
        raise signal

    with pytest.raises(signal):
        _run_a(_component(interrupted, ["f"]))


def test_a_failed_evaluation_is_not_cached() -> None:
    calls: list[float] = []

    def flaky(a: float) -> float:
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("first call fails")
        return a

    comp = _component(flaky, ["f"])

    with pytest.raises(ToolExecutionError):
        _run_a(comp)
    out = _run_a(comp)

    np.testing.assert_allclose(out["f"], [1.0])
    assert len(calls) == 2


# --- Derivatives ------------------------------------------------------------


def test_finite_difference_jacobian_is_available_out_of_the_box() -> None:
    comp = ToolComponent("quad", lambda x, y: x**2 + 3 * y, ["x", "y"], ["f"])

    comp.linearize({"x": _array(2.0), "y": _array(1.0)}, compute_all_jacobians=True)

    np.testing.assert_allclose(comp.jac["f"]["x"], [[4.0]], rtol=1e-4)
    np.testing.assert_allclose(comp.jac["f"]["y"], [[3.0]], rtol=1e-4)


def test_the_constructor_has_no_derivatives_option() -> None:
    with pytest.raises(TypeError):
        ToolComponent("t", _add, ["x", "y"], ["z"], derivatives=True)  # type: ignore[call-arg]


# --- Namespaces -------------------------------------------------------------


def test_a_namespaced_input_reaches_the_function_under_its_plain_name() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"])
    comp.add_namespace_to_input("x", "ns")

    out = comp.execute({"ns:x": _array(2.0), "y": _array(3.0)})

    np.testing.assert_allclose(out["z"], [5.0])


def test_a_namespaced_output_is_published_under_its_namespace() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"])
    comp.add_namespace_to_output("z", "ns")

    out = comp.execute({"x": _array(2.0), "y": _array(3.0)})

    np.testing.assert_allclose(out["ns:z"], [5.0])


# --- Caching ----------------------------------------------------------------


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, a: float) -> float:
        self.calls += 1
        return a


def test_a_deterministic_tool_is_called_once_for_the_same_input() -> None:
    counter = _Counter()
    comp = _component(counter, ["f"])

    for _ in range(5):
        _run_a(comp, 1.0)

    assert counter.calls == 1


def test_a_non_deterministic_tool_is_called_every_time() -> None:
    counter = _Counter()
    comp = _component(counter, ["f"], deterministic=False)

    for _ in range(5):
        _run_a(comp, 1.0)

    assert counter.calls == 5


def test_a_non_deterministic_tool_has_no_cache() -> None:
    assert _component(_Counter(), ["f"], deterministic=False).cache is None
    assert _component(_Counter(), ["f"]).cache is not None


# --- Argument names ---------------------------------------------------------


def test_arg_map_binds_one_function_to_different_graph_inputs() -> None:
    def scaled(a: float) -> float:
        return 10 * a

    on_x = ToolComponent("on_x", scaled, ["x"], ["fx"], arg_map={"x": "a"})
    on_y = ToolComponent("on_y", scaled, ["y"], ["fy"], arg_map={"y": "a"})

    np.testing.assert_allclose(on_x.execute({"x": _array(2.0)})["fx"], [20.0])
    np.testing.assert_allclose(on_y.execute({"y": _array(3.0)})["fy"], [30.0])


def test_inputs_absent_from_arg_map_keep_their_name() -> None:
    def func(a: float, y: float) -> float:
        return a - y

    comp = ToolComponent("t", func, ["x", "y"], ["z"], arg_map={"x": "a"})

    out = comp.execute({"x": _array(5.0), "y": _array(2.0)})

    np.testing.assert_allclose(out["z"], [3.0])


def test_the_graph_name_of_a_mapped_input_is_not_passed_to_the_function() -> None:
    comp = ToolComponent("t", lambda x: x, ["x"], ["z"], arg_map={"x": "a"})

    with pytest.raises(ToolExecutionError) as exc_info:
        comp.execute({"x": _array(1.0)})

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_the_arg_map_works_with_the_tool_value_decoding() -> None:
    specs = {"mode": {"name": "mode", "type": "choice", "values": ["a", "b"]}}
    seen: list[Any] = []

    def func(kind: str) -> float:
        seen.append(kind)
        return 1.0

    comp = ToolComponent(
        "t", func, ["mode"], ["z"], specs=specs, arg_map={"mode": "kind"}
    )
    comp.execute({"mode": _array(1.0)})

    assert seen == ["b"]


@pytest.mark.parametrize(
    ("inputs", "arg_map", "argument", "colliding"),
    [
        (["x", "y"], {"x": "y"}, "y", ["x", "y"]),
        (["x", "y"], {"x": "a", "y": "a"}, "a", ["x", "y"]),
        (["x", "y", "z"], {"x": "z", "y": "z"}, "z", ["x", "y", "z"]),
    ],
)
def test_two_inputs_for_the_same_argument_are_rejected(
    inputs: list[str],
    arg_map: dict[str, str],
    argument: str,
    colliding: list[str],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        ToolComponent("clash", lambda **kwargs: 1.0, inputs, ["f"], arg_map=arg_map)

    message = str(exc_info.value)
    assert "Tool 'clash'" in message
    assert f"'{argument}'" in message
    assert str(colliding) in message


def test_a_rename_to_a_free_argument_name_is_accepted() -> None:
    comp = ToolComponent(
        "swap", lambda x, y: x - y, ["x", "y"], ["f"], arg_map={"x": "y", "y": "x"}
    )

    out = comp.execute({"x": _array(5.0), "y": _array(2.0)})

    np.testing.assert_allclose(out["f"], [-3.0])


# --- Defaults ---------------------------------------------------------------


def test_default_input_data_holds_exactly_the_given_defaults() -> None:
    comp = ToolComponent(
        "add", _add, ["x", "y"], ["z"], defaults={"y": 2.0, "x": np.array([1.0])}
    )

    assert set(comp.default_input_data) == {"x", "y"}
    np.testing.assert_allclose(comp.default_input_data["y"], [2.0])
    np.testing.assert_allclose(comp.default_input_data["x"], [1.0])


def test_there_is_no_default_unless_one_is_given() -> None:
    assert dict(ToolComponent("add", _add, ["x", "y"], ["z"]).default_input_data) == {}


def test_a_defaulted_input_can_be_omitted() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"], defaults={"y": 2.0})

    out = comp.execute({"x": _array(1.0)})

    np.testing.assert_allclose(out["z"], [3.0])


def test_a_design_input_without_a_default_cannot_be_omitted() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"], defaults={"y": 2.0})

    with pytest.raises(InvalidDataError):
        comp.execute({})


def test_a_given_input_overrides_its_default() -> None:
    comp = ToolComponent("add", _add, ["x", "y"], ["z"], defaults={"y": 2.0})

    out = comp.execute({"x": _array(1.0), "y": _array(10.0)})

    np.testing.assert_allclose(out["z"], [11.0])


def test_the_constructor_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        ToolComponent("add", _add, ["x", "y"], ["z"], {"y": 2.0})  # type: ignore[misc]
