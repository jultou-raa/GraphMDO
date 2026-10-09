"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Shared fixtures for the end-to-end suite: real Ax + GEMSEO runs, no mocks.
Every Ax Client is seeded, so two runs of the suite give identical results.
"""

import logging
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from ax.api.client import Client

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.topology import TopologicalAnalyzer
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization import ax_algo_lib
from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.schema import (
    ConstraintSpec,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
)

SEED = 0
# The box centre x0 = (-3, 3) is far from the optimum.
SHIFTED_PARABOLOID_SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=-10.0, upper=4.0),
        RangeVar(name="y", lower=-4.0, upper=10.0),
        StateVar(name="f_xy"),
        StateVar(name="c_xy"),
    ],
    tools=[ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy", "c_xy"])],
)


def paraboloid(x: float, y: float) -> dict[str, float]:
    return {"f_xy": (x - 3) ** 2 + x * y + (y + 4) ** 2 - 3, "c_xy": x - y}


class RecordedTool:
    """Calls `function` and keeps the keyword inputs of every call, even failed ones."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **inputs: Any) -> Any:
        self.calls.append(inputs)
        return self.function(**inputs)


class AxRecorder:
    """Builds every Ax Client with a fixed seed and keeps it for inspection.

    This is the only code in the suite that reads private Ax state; #51 and #53
    will replace it with public API.
    """

    def __init__(self) -> None:
        self.clients: list[Client] = []

    def __call__(self, random_seed: int | None = None) -> Client:
        seed = SEED if random_seed is None else random_seed
        self.clients.append(Client(random_seed=seed))
        return self.clients[-1]

    def trial_kinds(self, index: int = -1) -> Counter[tuple[str, str]]:
        """Counts trials by (generation node, status); the attached x0 is "x0"."""
        trials = self.clients[index]._experiment.trials.values()
        return Counter(
            (
                trial.generator_runs[0]._generation_node_name or "x0",
                trial.status.name,
            )
            for trial in trials
        )

    def count(self, status: str, index: int = -1) -> int:
        kinds = self.trial_kinds(index)
        return sum(
            n for (_, trial_status), n in kinds.items() if trial_status == status
        )

    def objective_thresholds(self, index: int = -1) -> dict[str, float]:
        """The objective thresholds of the experiment, by metric name."""
        config = self.clients[index]._experiment.optimization_config
        return {t.metric.name: t.bound for t in config.objective_thresholds}


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)  # optimize() writes XDSM/plot files (#54)
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def ax_recorder(monkeypatch) -> AxRecorder:
    # AxOptimizationLibrary resolves the module global `Client` at construction.
    recorder = AxRecorder()
    monkeypatch.setattr(ax_algo_lib, "Client", recorder)
    return recorder


@pytest.fixture
def recorded() -> type[RecordedTool]:
    return RecordedTool


@pytest.fixture
def build_optimizer() -> Callable[..., tuple[BayesianOptimizer, LocalEvaluator]]:
    """Graph schema -> topology -> GEMSEO problem -> local evaluator -> optimizer."""

    def build(
        schema: StudySchema,
        registry: dict[str, Callable[..., Any]],
        objectives: list[dict[str, Any] | ObjectiveSpec],
        constraints: list[dict[str, Any] | ConstraintSpec] | None = None,
    ) -> tuple[BayesianOptimizer, LocalEvaluator]:
        objective_specs = [ObjectiveSpec.model_validate(o) for o in objectives]
        constraint_specs = [ConstraintSpec.model_validate(c) for c in constraints or []]
        outputs = [spec.name for spec in objective_specs + constraint_specs]
        analyzer = TopologicalAnalyzer(schema)
        resolved = analyzer.resolve_dependencies(outputs)
        builder = GraphProblemBuilder(schema)
        evaluator = LocalEvaluator(
            builder.build_problem(registry), builder.variable_specs
        )
        optimizer = BayesianOptimizer(
            evaluator,
            resolved.design_variables,
            objective_specs,
            constraint_specs,
        )
        return optimizer, evaluator

    return build


@pytest.fixture
def shifted_paraboloid(
    build_optimizer,
) -> Callable[..., tuple[BayesianOptimizer, RecordedTool]]:
    """Builds min f_xy subject to c_xy (op) bound, with c_xy = x - y.

    With c_xy <= 0 the optimum is (-1/3, -1/3), f* = 65/3; f(x0) = 73.
    """

    def build(
        op: str = "<=", bound: float = 0.0
    ) -> tuple[BayesianOptimizer, RecordedTool]:
        tool = RecordedTool(paraboloid)
        optimizer, _ = build_optimizer(
            SHIFTED_PARABOLOID_SCHEMA,
            {"Paraboloid": tool},
            [{"name": "f_xy", "minimize": True}],
            [{"name": "c_xy", "op": op, "bound": bound}],
        )
        return optimizer, tool

    return build
