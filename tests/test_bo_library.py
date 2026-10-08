"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from gemseo import create_scenario
from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.discipline import Discipline

from mdo_framework.core.components import ToolComponent
from mdo_framework.core.errors import InfeasiblePointError
from mdo_framework.core.topology import to_parameter_definition
from mdo_framework.optimization.bo_library import (
    BaseBOLibrary,
    add_constraints,
    build_metric_bindings,
)
from mdo_framework.optimization.bo_types import (
    BORunResult,
    Candidate,
    TrialOutcome,
)
from mdo_framework.optimization.errors import OptimizationConfigurationError
from mdo_framework.optimization.random_search import (
    RandomSearchLibrary,
    RandomSearchSettings,
)
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    DesignVariable,
    ObjectiveSpec,
    RangeVar,
)

X = RangeVar(name="x", lower=0.0, upper=10.0)
Y = RangeVar(name="y", lower=0.0, upper=10.0)
F = ObjectiveSpec(name="f")


class TransportError(RuntimeError):
    """Stands for a failure outside the tool, such as a lost service."""


class RawDiscipline(Discipline):
    """Discipline running a function on floats, with no output check."""

    def __init__(
        self, func: Callable[..., dict[str, float]], inputs: Sequence[str], outputs
    ) -> None:
        super().__init__(name="raw")
        self.input_grammar.update_from_names(inputs)
        self.output_grammar.update_from_names(outputs)
        self._inputs = tuple(inputs)
        self._func = func

    def _run(self, input_data):
        values = self._func(**{n: float(input_data[n][0]) for n in self._inputs})
        return {name: np.atleast_1d(float(value)) for name, value in values.items()}


def make_design_space(variables: Sequence[DesignVariable]) -> DesignSpace:
    space = DesignSpace()
    for variable in variables:
        if isinstance(variable, ChoiceVar):
            space.add_variable(
                variable.name,
                lower_bound=0,
                upper_bound=len(variable.choices) - 1,
                type_="integer",
                value=0,
            )
        else:
            kwargs = {"type_": "integer"} if variable.value_type == "int" else {}
            space.add_variable(
                variable.name,
                lower_bound=variable.lower,
                upper_bound=variable.upper,
                **kwargs,
            )
    return space


@dataclass
class Study:
    """A GEMSEO problem around a recorded tool, and the library to run on it."""

    problem: OptimizationProblem
    calls: list[dict[str, Any]]
    variables: tuple[DesignVariable, ...]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    library: RandomSearchLibrary
    parameter_constraints: tuple[str, ...] = field(default=())

    def settings(self, **options: Any) -> RandomSearchSettings:
        options.setdefault("enable_progress_bar", False)
        options.setdefault("log_problem", False)
        return RandomSearchSettings(
            design_variables=self.variables,
            objectives=self.objectives,
            constraints=self.constraints,
            parameter_constraints=self.parameter_constraints,
            **options,
        )

    def run(self, **options: Any) -> BORunResult:
        self.library.execute(self.problem, settings_model=self.settings(**options))
        assert self.library.result is not None
        return self.library.result

    def points(self, name: str) -> list[Any]:
        return [call[name] for call in self.calls]


def make_study(
    func: Callable[..., dict[str, float]],
    variables: Sequence[DesignVariable],
    objectives: Sequence[ObjectiveSpec] = (F,),
    constraints: Sequence[ConstraintSpec] = (),
    *,
    parameter_constraints: Sequence[str] = (),
    raw: bool = False,
    library: RandomSearchLibrary | None = None,
) -> Study:
    calls: list[dict[str, Any]] = []

    def recorded(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return func(**kwargs)

    names = [variable.name for variable in variables]
    outputs = list(
        dict.fromkeys([*(o.name for o in objectives), *(c.name for c in constraints)])
    )
    if raw:
        discipline = RawDiscipline(recorded, names, outputs)
    else:
        discipline = ToolComponent(
            "tool",
            recorded,
            names,
            outputs,
            specs={v.name: to_parameter_definition(v) for v in variables},
        )
    scenario = create_scenario(
        [discipline],
        formulation_name="MDF",
        objective_name=[o.name for o in objectives],
        design_space=make_design_space(variables),
        maximize_objective=len(objectives) == 1 and not objectives[0].minimize,
    )
    add_constraints(scenario, constraints)
    return Study(
        problem=scenario.formulation.optimization_problem,
        calls=calls,
        variables=tuple(variables),
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        library=library or RandomSearchLibrary(),
        parameter_constraints=tuple(parameter_constraints),
    )


def quadratic(x: float) -> dict[str, float]:
    return {"f": (x - 3.0) ** 2}


# Settings


def settings(variables=(X,), **options: Any) -> RandomSearchSettings:
    return RandomSearchSettings(design_variables=variables, objectives=(F,), **options)


@pytest.mark.parametrize(
    ("options", "variables", "expected"),
    [
        ({}, (X,), 15),
        ({"n_init": 3, "n_steps": 4}, (X,), 7),
        ({"n_init": 3, "n_steps": 4, "evaluate_x0": True}, (X,), 8),
        (
            {"n_init": 3, "n_steps": 4, "evaluate_x0": False},
            (X.model_copy(update={"initial": 2.0}),),
            7,
        ),
        ({"n_init": 3, "n_steps": 4}, (X.model_copy(update={"initial": 2.0}),), 8),
    ],
)
def test_max_iter_is_derived_from_the_phases(options, variables, expected):
    assert settings(variables, **options).max_iter == expected


def test_a_consistent_max_iter_is_accepted():
    assert settings(n_init=3, n_steps=4, max_iter=7).max_iter == 7


@pytest.mark.parametrize(
    "options",
    [
        {"n_init": 3, "n_steps": 4, "max_iter": 9},
        {"n_init": 3, "n_steps": 4, "evaluate_x0": True, "max_iter": 7},
    ],
)
def test_a_conflicting_max_iter_is_rejected(options):
    with pytest.raises(ValueError, match="max_iter"):
        settings(**options)


@pytest.mark.parametrize("options", [{"n_init": 0}, {"n_steps": 0}, {"batch_size": 0}])
def test_phase_sizes_must_be_positive(options):
    with pytest.raises(ValueError):
        settings(**options)


def test_a_study_needs_a_design_variable_and_an_objective():
    with pytest.raises(ValueError):
        RandomSearchSettings(design_variables=(), objectives=(F,))
    with pytest.raises(ValueError):
        RandomSearchSettings(design_variables=(X,), objectives=())


@pytest.mark.parametrize(
    ("evaluate_x0", "variables", "expected"),
    [
        (None, (X.model_copy(update={"initial": 1.0}),), True),
        (None, (X.model_copy(update={"initial": 1.0}), Y), False),
        (None, (X,), False),
        (True, (X,), True),
        (False, (X.model_copy(update={"initial": 1.0}),), False),
    ],
)
def test_x0_is_evaluated_if_asked_or_every_variable_has_an_initial(
    evaluate_x0, variables, expected
):
    assert settings(variables, evaluate_x0=evaluate_x0).x0_enabled is expected


def test_the_gemseo_hooks_used_by_the_driver_still_exist():
    assert callable(BaseDriverLibrary._check_stopping_criteria)
    assert callable(BaseDriverLibrary._init_iter_observer)
    assert callable(BaseDriverLibrary._finalize_previous_iteration)
    assert callable(BaseOptimizationLibrary._check_constraints_handling)
    assert callable(BaseOptimizationLibrary._pre_run)


def test_the_library_has_no_result_before_a_run():
    assert RandomSearchLibrary().result is None


# Budget


def test_the_budget_is_the_number_of_tool_calls():
    study = make_study(quadratic, [X])

    result = study.run(n_init=3, n_steps=4, seed=1)

    assert len(study.calls) == 7
    assert result.stop_reason == "budget"
    assert result.evaluations == {"x0": 0, "init": 3, "bo": 4, "failed": 0}
    assert [r.phase for r in result.records] == ["init"] * 3 + ["bo"] * 4
    assert [r.index for r in result.records] == list(range(7))


def test_x0_is_evaluated_first_at_the_start_point():
    study = make_study(quadratic, [X])

    result = study.run(n_init=3, n_steps=4, evaluate_x0=True, seed=1)

    assert len(study.calls) == 8
    assert study.calls[0] == {"x": 5.0}
    assert result.evaluations == {"x0": 1, "init": 3, "bo": 4, "failed": 0}
    assert result.records[0].phase == "x0"


def test_x0_is_evaluated_when_every_variable_declares_an_initial():
    study = make_study(quadratic, [X.model_copy(update={"initial": 2.0})])

    result = study.run(n_init=3, n_steps=4, seed=1)

    assert study.calls[0] == {"x": 2.0}
    assert len(study.calls) == 8
    assert result.records[0].phase == "x0"


def test_x0_is_skipped_when_only_some_variables_declare_an_initial():
    study = make_study(
        lambda x, y: {"f": x + y}, [X.model_copy(update={"initial": 2.0}), Y]
    )

    result = study.run(n_init=3, n_steps=4, seed=1)

    assert len(study.calls) == 7
    assert result.evaluations["x0"] == 0


def test_a_batch_does_not_exceed_the_budget():
    study = make_study(quadratic, [X])

    result = study.run(n_init=3, n_steps=4, batch_size=3, seed=1)

    assert len(study.calls) == 7
    assert result.stop_reason == "budget"
    assert result.evaluations == {"x0": 0, "init": 3, "bo": 4, "failed": 0}


def test_x0_is_the_declared_initial_for_every_variable_kind():
    variables = [
        X.model_copy(update={"initial": 2.5}),
        RangeVar(name="n", lower=1, upper=9, value_type="int", initial=4),
        ChoiceVar(name="mode", choices=["a", "b", "c"], initial="c"),
    ]
    study = make_study(lambda x, n, mode: {"f": x + n}, variables)

    study.run(n_init=2, n_steps=1, seed=1)

    assert study.calls[0] == {"x": 2.5, "n": 4, "mode": "c"}


# Stopping and plateaus (#43)


def test_a_penalty_plateau_does_not_stop_the_run():
    study = make_study(lambda x: {"f": 1e6 if x > 3.0 else (x - 1.0) ** 2}, [X])

    result = study.run(n_init=4, n_steps=8, seed=2)

    assert len(study.calls) == 12
    assert result.stop_reason == "budget"


def test_non_finite_outputs_are_failed_trials_and_use_the_whole_budget():
    study = make_study(
        lambda x: {"f": float("nan") if x > 3.0 else (x - 1.0) ** 2}, [X], raw=True
    )

    result = study.run(n_init=4, n_steps=8, max_consecutive_failures=12, seed=2)

    failed = [r for r in result.records if r.outcome.status == "failed"]
    assert len(study.calls) == 12
    assert result.stop_reason == "budget"
    assert failed
    assert all(r.outcome.reason.startswith("NON_FINITE_OUTPUT") for r in failed)
    assert result.best is not None
    assert result.best.parameters["x"] <= 3.0


def test_a_tool_returning_nan_is_a_failed_trial():
    study = make_study(lambda x: {"f": float("nan") if x > 3.0 else x}, [X], raw=False)

    result = study.run(n_init=4, n_steps=8, max_consecutive_failures=12, seed=2)

    failed = [r for r in result.records if r.outcome.status == "failed"]
    assert failed
    assert all(r.outcome.reason.startswith("OUTPUT_INVALID") for r in failed)


def test_the_run_stops_when_the_time_is_up():
    def slow(x):
        time.sleep(0.05)
        return {"f": x}

    study = make_study(slow, [X])

    result = study.run(n_init=5, n_steps=5, max_time=0.12, seed=1)

    assert result.stop_reason == "max_time"
    assert 1 <= len(study.calls) < 10
    assert result.records[-1].outcome.status == "abandoned"
    assert result.evaluations["failed"] == 0


class LateStartLibrary(RandomSearchLibrary):
    """Library whose start-up outlasts the time limit."""

    def _pre_run(self, problem):
        super()._pre_run(problem)
        time.sleep(0.05)


def test_the_time_is_up_before_x0():
    study = make_study(quadratic, [X], library=LateStartLibrary())

    result = study.run(n_init=3, n_steps=4, evaluate_x0=True, max_time=0.01)

    assert result.stop_reason == "max_time"
    assert study.calls == []
    assert [r.outcome.status for r in result.records] == ["abandoned"]
    assert result.records[0].phase == "x0"
    assert result.best is None


# Failures (#46)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("boom"), "TOOL_FAILED"),
        (KeyError("missing"), "TOOL_FAILED"),
        (OSError("disk"), "TOOL_FAILED"),
        (TypeError("bad"), "TOOL_FAILED"),
        (subprocess.CalledProcessError(1, "cmd"), "TOOL_FAILED"),
        (InfeasiblePointError("incomputable"), "POINT_INFEASIBLE"),
    ],
)
def test_a_failing_tool_is_a_failed_trial_and_the_run_goes_on(error, code):
    def tool(x):
        if x > 5.0:
            raise error
        return {"f": (x - 3.0) ** 2}

    study = make_study(tool, [X])

    result = study.run(n_init=4, n_steps=8, max_consecutive_failures=12, seed=3)

    failing = sum(x > 5.0 for x in study.points("x"))
    failed = [r for r in result.records if r.outcome.status == "failed"]
    assert 0 < failing < 12
    assert len(study.calls) == 12
    assert result.stop_reason == "budget"
    assert len(failed) == failing
    assert all(r.outcome.reason.startswith(f"{code}: ") for r in failed)
    assert all(r.parameters["x"] > 5.0 for r in failed)
    assert result.best is not None


def test_a_tool_failing_everywhere_stops_the_run():
    def tool(x):
        raise RuntimeError("always")

    study = make_study(tool, [X])

    result = study.run(n_init=5, n_steps=10, max_consecutive_failures=3, seed=1)

    assert len(study.calls) == 3
    assert result.stop_reason == "consecutive_failures"
    assert result.best is None
    assert result.evaluations["failed"] == 3


def test_a_tool_failing_everywhere_stops_with_the_progress_bar_on():
    def tool(x):
        raise RuntimeError("always")

    study = make_study(tool, [X])

    result = study.run(
        n_init=5,
        n_steps=10,
        max_consecutive_failures=3,
        seed=1,
        enable_progress_bar=True,
    )

    assert result.stop_reason == "consecutive_failures"


def test_a_success_resets_the_count_of_consecutive_failures():
    def tool(x):
        if int(x * 10) % 2 == 0:
            raise RuntimeError("half of the points")
        return {"f": x}

    study = make_study(tool, [X])

    result = study.run(n_init=10, n_steps=10, max_consecutive_failures=8, seed=4)

    assert result.stop_reason == "budget"
    assert len(study.calls) == 20


def test_an_error_outside_the_tool_aborts_and_keeps_the_trials_so_far():
    def tool(x):
        if len(study.calls) == 3:
            raise TransportError("service lost")
        return {"f": x}

    study = make_study(tool, [X], raw=True)

    with pytest.raises(TransportError, match="service lost"):
        study.run(n_init=5, n_steps=5, seed=1)

    result = study.library.result
    assert result is not None
    assert result.stop_reason == "aborted"
    assert [r.outcome.status for r in result.records] == ["completed", "completed"]


# Start point and parameter constraints (#48)


def test_declared_initials_violating_a_parameter_constraint_fail_before_any_call():
    study = make_study(
        lambda x, y: {"f": x + y},
        [X.model_copy(update={"initial": 8.0}), Y.model_copy(update={"initial": 8.0})],
        parameter_constraints=["x + y <= 10"],
    )

    with pytest.raises(OptimizationConfigurationError, match="x \\+ y <= 10"):
        study.run(n_init=3, n_steps=4, seed=1)

    assert study.calls == []


def test_the_start_and_every_candidate_satisfy_the_parameter_constraints():
    study = make_study(
        lambda x, y: {"f": x + y},
        [X, Y],
        parameter_constraints=["x + y <= 4"],
    )

    result = study.run(n_init=10, n_steps=10, evaluate_x0=True, seed=1)

    assert len(study.calls) == 21
    assert result.records[0].phase == "x0"
    assert all(call["x"] + call["y"] <= 4.0 + 1e-6 for call in study.calls)


def test_a_tool_failing_at_x0_is_a_failed_x0_and_the_run_goes_on():
    def tool(x):
        if x == 5.0:
            raise RuntimeError("bad start")
        return {"f": x}

    study = make_study(tool, [X])

    result = study.run(n_init=3, n_steps=4, evaluate_x0=True, seed=1)

    assert len(study.calls) == 8
    assert result.records[0].phase == "x0"
    assert result.records[0].outcome.status == "failed"
    assert result.stop_reason == "budget"
    assert result.best is not None


def test_a_constraint_that_no_draw_satisfies_is_a_configuration_error():
    study = make_study(
        lambda x, y: {"f": x + y}, [X, Y], parameter_constraints=["x + y <= 0.00001"]
    )

    with pytest.raises(OptimizationConfigurationError, match="parameter constraints"):
        study.run(n_init=3, n_steps=4, seed=1)

    assert study.library.result.stop_reason == "aborted"


# Directions, names, units (#34 #35 #49 #50)


def test_maximisation_reports_raw_values_under_the_user_name():
    study = make_study(
        lambda x: {"f": 10.0 - (x - 4.0) ** 2},
        [X],
        [ObjectiveSpec(name="f", minimize=False)],
    )

    result = study.run(n_init=5, n_steps=10, seed=3)

    completed = [r for r in result.records if r.outcome.status == "completed"]
    for record in completed:
        assert record.outcome.metrics == {
            "f": pytest.approx(10.0 - (record.parameters["x"] - 4.0) ** 2)
        }
        entry = record.to_history_entry(result.bindings)
        assert set(entry["objectives"]) == {"f"}
    assert [b.name for b in result.bindings] == ["f"]
    assert result.best.outcome.metrics["f"] == max(
        r.outcome.metrics["f"] for r in completed
    )
    assert result.best.outcome.metrics["f"] > 5.0


def test_multi_objective_with_mixed_directions_has_a_consistent_front():
    objectives = [
        ObjectiveSpec(name="f1", minimize=True),
        ObjectiveSpec(name="f2", minimize=False),
    ]
    study = make_study(lambda x, y: {"f1": x, "f2": y}, [X, Y], objectives)

    result = study.run(n_init=10, n_steps=15, seed=5)

    completed = [r for r in result.records if r.outcome.status == "completed"]

    def dominates(a, b):
        ma, mb = a.outcome.metrics, b.outcome.metrics
        return (
            ma["f1"] <= mb["f1"]
            and ma["f2"] >= mb["f2"]
            and (ma["f1"] < mb["f1"] or ma["f2"] > mb["f2"])
        )

    expected = [
        r.index for r in completed if not any(dominates(o, r) for o in completed)
    ]
    assert [r.index for r in result.pareto_front] == expected
    assert len(expected) > 1
    assert result.best in result.pareto_front
    entry = result.best.to_history_entry(result.bindings)
    assert set(entry["objectives"]) == {"f1", "f2"}
    assert result.best.outcome.metrics == {
        "f1": result.best.parameters["x"],
        "f2": result.best.parameters["y"],
    }


def test_binding_signs_and_offsets_undo_the_gemseo_transformations():
    constraints = (
        ConstraintSpec(name="upper", bound=4.0, op="<=", tolerance=0.5),
        ConstraintSpec(name="lower", bound=3.0, op=">=", tolerance=0.5),
    )
    study = make_study(
        lambda x: {"f": x, "upper": x, "lower": x},
        [X],
        [ObjectiveSpec(name="f", minimize=False)],
        constraints,
    )

    f, upper, lower = build_metric_bindings(
        study.problem, study.objectives, study.constraints
    )

    assert (f.name, f.role, f.sign, f.minimize) == ("f", "objective", -1.0, False)
    assert (upper.sign, upper.offset, upper.op) == (1.0, 4.5, "<=")
    assert (lower.sign, lower.offset, lower.op) == (-1.0, 2.5, ">=")
    assert upper.tolerance == lower.tolerance == 0.5


def test_a_problem_that_does_not_match_the_specs_is_a_configuration_error():
    constraint = ConstraintSpec(name="c", bound=1.0)
    study = make_study(lambda x: {"f": x, "c": x}, [X], [F], [constraint])

    with pytest.raises(OptimizationConfigurationError, match="objective"):
        build_metric_bindings(study.problem, (ObjectiveSpec(name="g"),), (constraint,))
    with pytest.raises(OptimizationConfigurationError, match="constraint"):
        build_metric_bindings(study.problem, (F,), ())
    with pytest.raises(OptimizationConfigurationError, match="constraint"):
        build_metric_bindings(
            study.problem, (F,), (ConstraintSpec(name="other", bound=1.0),)
        )


def test_feasibility_agrees_with_gemseo_for_a_ge_constraint_with_tolerance():
    constraint = ConstraintSpec(name="c", bound=3.0, op=">=", tolerance=0.5)
    study = make_study(
        lambda x: {"f": (x - 1.0) ** 2, "c": x},
        [X.model_copy(update={"initial": 2.8})],
        [F],
        [constraint],
    )

    result = study.run(n_init=8, n_steps=12, seed=6)

    x0 = result.records[0]
    assert x0.phase == "x0"
    assert x0.feasible(result.bindings)
    entry = x0.to_history_entry(result.bindings)
    assert entry["constraints"]["c"]["value"] == pytest.approx(2.8)
    assert entry["constraints"]["c"]["margin"] == pytest.approx(-0.2)
    flags = []
    for record in result.records:
        outputs, _ = study.problem.evaluate_functions(
            np.array([record.parameters["x"]]), design_vector_is_normalized=False
        )
        gemseo_violated = any(
            np.any(np.asarray(outputs[c.name]) > 0.0) for c in study.problem.constraints
        )
        assert record.feasible(result.bindings) is (not gemseo_violated)
        flags.append(record.feasible(result.bindings))
    assert any(flags) and not all(flags)


@pytest.mark.parametrize(
    ("bound", "op", "feasible"), [(4.0, "<=", True), (12.0, ">=", False)]
)
def test_the_selected_design_does_not_depend_on_the_unit(bound, op, feasible):
    def select(k: float):
        x = RangeVar(name="x", lower=0.0, upper=10.0 * k)
        constraint = ConstraintSpec(
            name="c", bound=bound * k, op=op, tolerance=0.5 * k, scale=k
        )
        study = make_study(
            lambda x: {"f": (x - 3.0 * k) ** 2, "c": x}, [x], [F], [constraint]
        )
        result = study.run(n_init=8, n_steps=12, seed=7)
        return result, [r.parameters["x"] / k for r in result.records]

    in_metres, draws_m = select(1.0)
    in_millimetres, draws_mm = select(1000.0)

    assert draws_mm == pytest.approx(draws_m)
    assert in_metres.feasible is in_millimetres.feasible is feasible
    assert in_millimetres.best.parameters["x"] / 1000.0 == pytest.approx(
        in_metres.best.parameters["x"]
    )


# Search space


@pytest.mark.parametrize(
    "variable",
    [
        ChoiceVar(name="x", choices=["a", "b", "c"]),
        RangeVar(name="x", lower=1, upper=3, value_type="int"),
    ],
)
def test_an_exhausted_discrete_space_stops_without_duplicate_calls(variable):
    study = make_study(lambda x: {"f": float(len(str(x)))}, [variable])

    result = study.run(n_init=20, n_steps=20, seed=1)

    assert result.stop_reason == "search_space_exhausted"
    assert len(study.calls) == 3
    assert len({str(x) for x in study.points("x")}) == 3
    assert result.evaluations["init"] == 3


def test_a_log_range_is_sampled_on_a_log_scale():
    variable = RangeVar(name="x", lower=1e-3, upper=1e3, scaling="log")
    study = make_study(lambda x: {"f": x}, [variable])

    study.run(n_init=60, n_steps=60, seed=5)

    xs = np.array(study.points("x"))
    assert xs.min() >= 1e-3 and xs.max() <= 1e3
    assert 0.3 < np.mean(xs < 1.0) < 0.7


def test_the_same_seed_gives_the_same_points():
    first = make_study(quadratic, [X])
    second = make_study(quadratic, [X])
    other = make_study(quadratic, [X])

    first.run(n_init=3, n_steps=4, seed=11)
    second.run(n_init=3, n_steps=4, seed=11)
    other.run(n_init=3, n_steps=4, seed=12)

    assert first.points("x") == second.points("x")
    assert first.points("x") != other.points("x")


# The driver loop, with a scripted backend


class ScriptedLibrary(RandomSearchLibrary):
    """Backend proposing scripted points and recording what it is told."""

    def __init__(self, points: Sequence[dict[str, float]], extra: int = 0) -> None:
        super().__init__()
        self.points = list(points)
        self.extra = extra
        self.prior: Sequence = ()
        self.asked: list[int] = []
        self.told: list[tuple[Candidate, TrialOutcome]] = []

    def _setup(self, space, bindings, prior):
        self.prior = prior

    def _ask(self, n):
        self.asked.append(n)
        batch = self.points[: n + self.extra]
        del self.points[: n + self.extra]
        return [Candidate(point, key=i) for i, point in enumerate(batch)]

    def _tell(self, candidate, outcome):
        self.told.append((candidate, outcome))


def test_a_repeated_proposal_is_told_but_neither_evaluated_nor_recorded():
    library = ScriptedLibrary([{"x": 2.0}] * 8)
    study = make_study(quadratic, [X], library=library)

    result = study.run(n_init=5, n_steps=5)

    assert len(study.calls) == 1
    assert len(result.records) == 1
    assert result.stop_reason == "search_space_exhausted"
    assert len(library.told) == 6
    assert all(outcome == result.records[0].outcome for _, outcome in library.told)


def test_the_backend_is_set_up_with_the_x0_record():
    library = ScriptedLibrary([{"x": 2.0}])
    study = make_study(quadratic, [X], library=library)

    study.run(n_init=1, n_steps=1, evaluate_x0=True)

    assert [r.phase for r in library.prior] == ["x0"]
    assert library.prior[0].parameters == {"x": 5.0}


def test_the_backend_is_never_told_about_nan():
    library = ScriptedLibrary([{"x": 2.0}, {"x": 8.0}])
    study = make_study(
        lambda x: {"f": float("nan") if x > 5.0 else x}, [X], raw=True, library=library
    )

    study.run(n_init=1, n_steps=1)

    statuses = [outcome.status for _, outcome in library.told]
    assert statuses == ["completed", "failed"]
    assert library.told[1][1].metrics == {}
    assert library.told[1][1].reason.startswith("NON_FINITE_OUTPUT")


def test_candidates_beyond_the_requested_count_are_abandoned():
    library = ScriptedLibrary([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}], extra=1)
    study = make_study(quadratic, [X], library=library)

    result = study.run(n_init=1, n_steps=1)

    statuses = [r.outcome.status for r in result.records]
    assert statuses == ["completed", "abandoned", "completed", "abandoned"]
    assert result.stop_reason == "budget"
    assert study.calls == [{"x": 1.0}, {"x": 3.0}]
    assert library.asked == [1, 1]
    assert [outcome.status for _, outcome in library.told] == statuses
    assert result.records[1].outcome.reason == "surplus"


def test_a_repeated_failed_proposal_is_told_but_neither_evaluated_nor_recorded():
    def tool(x):
        raise RuntimeError("always")

    library = ScriptedLibrary([{"x": 2.0}] * 8)
    study = make_study(tool, [X], library=library)

    result = study.run(n_init=5, n_steps=5, max_consecutive_failures=5)

    assert len(study.calls) == 1
    assert len(result.records) == 1
    assert result.stop_reason == "search_space_exhausted"
    assert len(library.told) == 6
    assert all(outcome == result.records[0].outcome for _, outcome in library.told)


def test_a_proposal_repeating_x0_is_told_the_x0_outcome():
    library = ScriptedLibrary([{"x": 5.0}, {"x": 2.0}])
    study = make_study(quadratic, [X], library=library)

    result = study.run(n_init=1, n_steps=1, evaluate_x0=True)

    assert study.calls == [{"x": 5.0}, {"x": 2.0}]
    assert [r.phase for r in result.records] == ["x0", "init"]
    assert library.told[0][1] == result.records[0].outcome


def test_the_rest_of_a_batch_is_abandoned_after_too_many_failures():
    def tool(x):
        raise RuntimeError("always")

    library = ScriptedLibrary([{"x": float(i)} for i in range(1, 5)])
    study = make_study(tool, [X], library=library)

    result = study.run(n_init=3, n_steps=3, batch_size=4, max_consecutive_failures=2)

    statuses = [r.outcome.status for r in result.records]
    assert statuses == ["failed", "failed", "abandoned", "abandoned"]
    assert result.stop_reason == "consecutive_failures"
    assert len(study.calls) == 2
    assert library.asked == [4]


def test_the_driver_is_abstract():
    assert BaseBOLibrary.__abstractmethods__ >= {"_setup", "_ask", "_tell"}
