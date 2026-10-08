"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from ax.api.client import Client
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.types import ComparisonOp
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace

from mdo_framework.core.components import ToolComponent
from mdo_framework.core.errors import InfeasiblePointError
from mdo_framework.core.topology import to_parameter_definition
from mdo_framework.optimization import ax_algo_lib
from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary, AxSettings
from mdo_framework.optimization.bo_library import add_constraints
from mdo_framework.optimization.bo_types import BOSpace, BORunResult, MetricBinding
from mdo_framework.optimization.errors import OptimizationExecutionError
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    DesignVariable,
    ObjectiveSpec,
    RangeVar,
)

SEED = 11
X = RangeVar(name="x", lower=0.0, upper=10.0)
Y = RangeVar(name="y", lower=0.0, upper=10.0)
F = ObjectiveSpec(name="f")


class RecordingFactory:
    """Client factory building real seeded Ax clients, and keeping them."""

    def __init__(self, build: Callable[..., Client] = Client) -> None:
        self.build = build
        self.seeds: list[int | None] = []

    def __call__(self, random_seed: int | None = None) -> Client:
        self.seeds.append(random_seed)
        return self.build(random_seed=random_seed)


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
class Run:
    """A finished run of the Ax library on a recorded tool."""

    result: BORunResult
    library: AxOptimizationLibrary
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def client(self) -> Client:
        return self.library._client

    @property
    def experiment(self) -> Any:
        return self.client._experiment

    @property
    def config(self) -> Any:
        return self.experiment.optimization_config

    def kinds(self) -> list[tuple[str, str]]:
        """Return the (generation node, status) of every Ax trial."""
        return [
            (trial.generator_runs[0]._generation_node_name or "x0", trial.status.name)
            for trial in self.experiment.trials.values()
        ]

    def summary(self) -> Any:
        """Return the Ax summary table of the trials."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return self.client.summarize()

    def metric(self, name: str) -> dict[int, float]:
        """Return the value Ax holds for a metric, by trial index."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            frame = self.experiment.lookup_data().df
        rows = frame[frame["metric_name"] == name]
        return dict(zip(rows["trial_index"], rows["mean"], strict=True))

    def generator_kwargs(self) -> dict[str, Any]:
        """Return the fixed arguments of the BoTorch generator."""
        node = self.client._generation_strategy._nodes[1]
        return node.generator_spec_to_gen_from.generator_kwargs


def run_study(
    func: Callable[..., dict[str, float]],
    variables: Sequence[DesignVariable] = (X,),
    objectives: Sequence[ObjectiveSpec] = (F,),
    constraints: Sequence[ConstraintSpec] = (),
    *,
    parameter_constraints: Sequence[str] = (),
    library: AxOptimizationLibrary | None = None,
    **options: Any,
) -> Run:
    calls: list[dict[str, Any]] = []

    def recorded(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return func(**kwargs)

    names = [variable.name for variable in variables]
    outputs = list(
        dict.fromkeys([*(o.name for o in objectives), *(c.name for c in constraints)])
    )
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
    library = library or AxOptimizationLibrary()
    options.setdefault("seed", SEED)
    options.setdefault("n_init", 2)
    options.setdefault("n_steps", 1)
    settings = AxSettings(
        design_variables=tuple(variables),
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        parameter_constraints=tuple(parameter_constraints),
        enable_progress_bar=False,
        log_problem=False,
        **options,
    )
    library.execute(scenario.formulation.optimization_problem, settings_model=settings)
    assert library.result is not None
    return Run(library.result, library, calls)


def quadratic(x: float) -> dict[str, float]:
    return {"f": (x - 3.0) ** 2}


# Search space


def test_parameters_are_configured_in_user_values():
    variables = (
        RangeVar(name="n", lower=1, upper=8, value_type="int"),
        RangeVar(name="r", lower=1e-3, upper=1e3, scaling="log"),
        ChoiceVar(name="k", choices=[1, 2, 4], ordered=True),
        ChoiceVar(name="flag", choices=[True, False]),
        ChoiceVar(name="mode", choices=["a", "b", "c"], ordered=False),
        ChoiceVar(name="tag", choices=["p", "q", "r"]),
        ChoiceVar(name="level", choices=[1.5, 2.5, 3.5]),
    )

    def tool(n, r, k, flag, mode, tag, level):
        return {"f": n + math.log(r) + k + int(flag) + len(mode + tag) + level}

    run = run_study(tool, variables)

    parameters = run.experiment.search_space.parameters
    assert list(parameters) == ["n", "r", "k", "flag", "mode", "tag", "level"]
    n, r, k, flag, mode, tag, level = (parameters[v.name] for v in variables)
    assert isinstance(n, RangeParameter)
    assert (n.parameter_type, n.lower, n.upper) == (ParameterType.INT, 1, 8)
    assert (r.parameter_type, r.log_scale) == (ParameterType.FLOAT, True)
    assert isinstance(k, ChoiceParameter)
    assert (k.parameter_type, k.values, k.is_ordered) == (
        ParameterType.INT,
        [1, 2, 4],
        True,
    )
    assert (flag.parameter_type, set(flag.values)) == (
        ParameterType.BOOL,
        {True, False},
    )
    assert (mode.parameter_type, mode.values, mode.is_ordered) == (
        ParameterType.STRING,
        ["a", "b", "c"],
        False,
    )
    assert all(isinstance(call["n"], int) for call in run.calls)
    assert all(isinstance(call["flag"], bool) for call in run.calls)
    assert all(call["mode"] in ("a", "b", "c") for call in run.calls)
    # the default order is the one of Ax: unordered for several strings only
    assert [flag.is_ordered, tag.is_ordered, level.is_ordered] == [True, False, True]


def test_parameter_constraints_are_enforced_by_ax():
    run = run_study(
        lambda x, y: {"f": x + y},
        (X, Y),
        parameter_constraints=("x + y <= 5",),
        n_init=3,
    )

    assert len(run.experiment.search_space.parameter_constraints) == 1
    assert len(run.calls) == 4
    assert all(call["x"] + call["y"] <= 5 + 1e-6 for call in run.calls)


# Generation strategy


def test_the_budget_goes_to_sobol_then_botorch():
    run = run_study(quadratic, n_init=2, n_steps=2)

    assert run.result.stop_reason == "budget"
    assert run.kinds() == [
        ("GenerationStep_0_Sobol", "COMPLETED"),
        ("GenerationStep_0_Sobol", "COMPLETED"),
        ("GenerationStep_1_BoTorch", "COMPLETED"),
        ("GenerationStep_1_BoTorch", "COMPLETED"),
    ]


def test_a_single_objective_hard_codes_the_acquisition():
    run = run_study(quadratic)

    assert run.generator_kwargs() == {
        "botorch_acqf_class": qLogNoisyExpectedImprovement
    }


def test_bonsai_leaves_the_acquisition_to_ax(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger=ax_algo_lib.logger.name):
        run = run_study(quadratic, use_bonsai=True)

    assert "BONSAI" in caplog.text
    assert run.client._generation_strategy.name == "bonsai"
    assert run.generator_kwargs() == {}
    assert run.kinds()[-1] == ("GenerationStep_1_BoTorch", "COMPLETED")


# Metrics


def test_a_maximised_objective_keeps_its_user_name_and_value():
    run = run_study(
        lambda x: {"f": -((x - 3.0) ** 2)},
        objectives=(ObjectiveSpec(name="f", minimize=False),),
    )

    assert isinstance(run.config, OptimizationConfig)
    assert (run.config.objective.metric.name, run.config.objective.minimize) == (
        "f",
        False,
    )
    assert sorted(run.metric("f").values()) == pytest.approx(
        sorted(-((call["x"] - 3.0) ** 2) for call in run.calls)
    )


def test_multi_objective_has_directions_thresholds_and_ax_acquisition():
    objectives = (
        ObjectiveSpec(name="f1", minimize=True, threshold=20.0),
        ObjectiveSpec(name="f2", minimize=False, threshold=-5.0),
    )

    run = run_study(
        lambda x: {"f1": (x - 1.0) ** 2, "f2": -((x - 4.0) ** 2)},
        objectives=objectives,
        n_init=4,
    )

    assert isinstance(run.config, MultiObjectiveOptimizationConfig)
    assert [(o.metric.name, o.minimize) for o in run.config.objective.objectives] == [
        ("f1", True),
        ("f2", False),
    ]
    assert [
        (t.metric.name, t.bound, t.op) for t in run.config.objective_thresholds
    ] == [("f1", 20.0, ComparisonOp.LEQ), ("f2", -5.0, ComparisonOp.GEQ)]
    assert run.generator_kwargs() == {}


def test_a_ge_constraint_keeps_its_user_name_and_is_scaled():
    constraint = ConstraintSpec(name="g", op=">=", bound=4.0, scale=2.0)

    run = run_study(lambda x: {"f": x, "g": x**2}, constraints=(constraint,))

    (outcome_constraint,) = run.config.outcome_constraints
    assert outcome_constraint.metric.name == "g"
    assert outcome_constraint.op == ComparisonOp.GEQ
    assert outcome_constraint.bound == pytest.approx(2.0)
    assert sorted(run.metric("g").values()) == pytest.approx(
        sorted(call["x"] ** 2 / 2.0 for call in run.calls)
    )
    assert sorted(run.metric("f").values()) == pytest.approx(
        sorted(call["x"] for call in run.calls)
    )


def test_a_le_constraint_defaults_to_an_unscaled_bound():
    run = run_study(
        lambda x: {"f": x, "g": x - 5.0},
        constraints=(ConstraintSpec(name="g", bound=1.0),),
    )

    (outcome_constraint,) = run.config.outcome_constraints
    assert (outcome_constraint.op, outcome_constraint.bound) == (ComparisonOp.LEQ, 1.0)


# Trials


def test_x0_is_attached_as_the_baseline_with_its_outcome():
    variables = (
        RangeVar(name="n", lower=1, upper=8, value_type="int", initial=3),
        RangeVar(name="x", lower=0.0, upper=10.0, initial=2.0),
    )

    run = run_study(lambda n, x: {"f": (x - 3.0) ** 2 + n}, variables, evaluate_x0=True)

    baseline = run.experiment.trials[0]
    assert baseline.arm.name == "baseline"
    assert baseline.arm.parameters == {"n": 3, "x": 2.0}
    assert run.kinds()[0] == ("x0", "COMPLETED")
    assert run.metric("f")[0] == pytest.approx(4.0)
    assert [kind for kind, _ in run.kinds()[1:]] == [
        "GenerationStep_0_Sobol",
        "GenerationStep_0_Sobol",
        "GenerationStep_1_BoTorch",
    ]


def test_a_failed_x0_is_attached_as_failed_and_the_run_goes_on():
    def tool(x):
        if x == 2.0:
            raise InfeasiblePointError("x0 is not computable")
        return quadratic(x)

    run = run_study(tool, (X.model_copy(update={"initial": 2.0}),), evaluate_x0=True)

    assert run.result.stop_reason == "budget"
    assert run.kinds()[0] == ("x0", "FAILED")
    assert [status for _, status in run.kinds()[1:]] == ["COMPLETED"] * 3
    reason = run.summary()["status_reason"][0]
    assert reason == run.result.records[0].outcome.reason
    assert "x0 is not computable" in reason


def test_failed_trials_are_marked_failed_in_ax_with_the_reason():
    def tool(x):
        if x > 6.0:
            raise InfeasiblePointError("above the limit")
        return quadratic(x)

    run = run_study(tool, n_init=5, n_steps=2)

    failed = [r for r in run.result.records if r.outcome.status == "failed"]
    assert failed
    summary = run.summary()
    ax_failed = summary[summary["trial_status"] == "FAILED"]
    assert list(ax_failed["status_reason"]) == [r.outcome.reason for r in failed]
    assert all("above the limit" in reason for reason in ax_failed["status_reason"])
    assert len(ax_failed) == run.result.evaluations["failed"]


def test_a_candidate_cut_by_the_stop_is_marked_abandoned_in_ax():
    def tool(x):
        raise InfeasiblePointError("never computable")

    run = run_study(tool, batch_size=3, n_init=3, n_steps=2, max_consecutive_failures=1)

    assert run.result.stop_reason == "consecutive_failures"
    assert run.kinds() == [
        ("GenerationStep_0_Sobol", "FAILED"),
        ("GenerationStep_0_Sobol", "ABANDONED"),
        ("GenerationStep_0_Sobol", "ABANDONED"),
    ]
    assert len(run.calls) == 1


def test_asking_before_any_outcome_is_reported():
    library = AxOptimizationLibrary()
    library._settings = AxSettings(
        design_variables=(X,), objectives=(F,), n_init=1, seed=SEED
    )
    binding = MetricBinding(
        name="f", role="objective", gemseo_name="f", index=0, sign=1.0
    )
    library._setup(BOSpace((X,)), (binding,), ())
    assert len(library._ask(1)) == 1

    with pytest.raises(OptimizationExecutionError, match="before the initial trials"):
        library._ask(1)


# Client


def test_the_seed_reaches_the_client_factory():
    factory = RecordingFactory()

    run_study(quadratic, library=AxOptimizationLibrary(client_factory=factory), seed=7)

    assert factory.seeds == [7]


def test_the_default_factory_is_the_client_of_the_module(
    monkeypatch: pytest.MonkeyPatch,
):
    factory = RecordingFactory(ax_algo_lib.Client)
    monkeypatch.setattr(ax_algo_lib, "Client", factory)

    run = run_study(quadratic, seed=5)

    assert factory.seeds == [5]
    assert run.result.stop_reason == "budget"


def test_the_same_seed_gives_the_same_run():
    first = run_study(quadratic, n_steps=2)
    second = run_study(quadratic, n_steps=2)

    assert first.calls == second.calls
