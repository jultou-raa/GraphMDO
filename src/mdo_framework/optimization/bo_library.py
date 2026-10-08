"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import math
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.evaluation_counter import EvaluationCounter
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.stop_criteria import MaxIterReachedException, MaxTimeReached
from pydantic import Field, NonNegativeFloat, PositiveInt, model_validator

from mdo_framework.core.errors import EvaluationError
from mdo_framework.optimization.bo_types import (
    BORunResult,
    BOSpace,
    Candidate,
    MetricBinding,
    Phase,
    StopReason,
    TrialOutcome,
    TrialRecord,
    gemseo_bound,
)
from mdo_framework.optimization.errors import OptimizationConfigurationError
from mdo_framework.schema import ConstraintSpec, DesignVariable, ObjectiveSpec
from mdo_framework.validation import (
    ParameterConstraintError,
    parse_parameter_constraints,
)


class BaseBOSettings(BaseOptimizerSettings):
    """Settings shared by the backends of the Bayesian optimization driver.

    The budget is set by the phases, never by ``max_iter``: the run makes
    ``n_init`` initial trials and ``n_steps`` model-driven trials, after the
    start point x0 when it is evaluated. ``max_iter`` is derived from them and
    only accepted if it agrees. The GEMSEO tolerances ``ftol_*``, ``xtol_*``
    and ``scaling_threshold`` are ignored: only the budget and ``max_time``
    stop a run, so a plateau never ends it early.

    Attributes:
        design_variables: Design variables, in the order of the design space.
        objectives: Objectives, in the order of the objective function.
        constraints: Inequality constraints, in the order they were added.
        parameter_constraints: Linear constraints between design variables.
        n_init: Number of initial trials.
        n_steps: Number of model-driven trials.
        evaluate_x0: Whether to evaluate x0 first. ``None`` evaluates it when
            every design variable declares an ``initial`` value.
        seed: Seed of the backend, ``None`` for a random one.
        batch_size: Maximum number of candidates asked at once.
        max_consecutive_failures: Failed trials in a row that end the run.
    """

    design_variables: tuple[DesignVariable, ...] = Field(min_length=1)
    objectives: tuple[ObjectiveSpec, ...] = Field(min_length=1)
    constraints: tuple[ConstraintSpec, ...] = ()
    parameter_constraints: tuple[str, ...] = ()
    n_init: PositiveInt = 5
    n_steps: PositiveInt = 10
    evaluate_x0: bool | None = None
    seed: int | None = None
    batch_size: PositiveInt = 1
    max_consecutive_failures: PositiveInt = 5
    ineq_tolerance: NonNegativeFloat = 0.0
    normalize_design_space: bool = False
    max_iter: PositiveInt = 1

    @property
    def x0_enabled(self) -> bool:
        """Whether x0 is evaluated."""
        if self.evaluate_x0 is not None:
            return self.evaluate_x0
        return all(variable.initial is not None for variable in self.design_variables)

    @model_validator(mode="after")
    def _derive_max_iter(self) -> "BaseBOSettings":
        budget = int(self.x0_enabled) + self.n_init + self.n_steps
        if "max_iter" in self.model_fields_set and self.max_iter != budget:
            raise ValueError(
                f"max_iter={self.max_iter} contradicts the phases, which make "
                f"{budget} evaluations: drop max_iter and set n_init and n_steps"
            )
        self.max_iter = budget
        return self


def build_metric_bindings(
    problem: OptimizationProblem,
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[ConstraintSpec],
) -> tuple[MetricBinding, ...]:
    """Bind every user output to the GEMSEO function that computes it.

    Args:
        problem: GEMSEO problem built from the study.
        objectives: Objective declarations, in the order of the objective.
        constraints: Constraint declarations, in the order they were added.

    Returns:
        The objective bindings followed by the constraint bindings.

    Raises:
        OptimizationConfigurationError: If the problem does not hold exactly
            these objectives and constraints.
    """
    objective = problem.objective
    names = [spec.name for spec in objectives]
    if list(objective.output_names) != names:
        raise OptimizationConfigurationError(
            f"the objective of the problem computes {list(objective.output_names)}, "
            f"but the study declares the objectives {names}"
        )
    if len(problem.constraints) != len(constraints):
        raise OptimizationConfigurationError(
            f"the problem has {len(problem.constraints)} constraints, "
            f"but the study declares {len(constraints)}"
        )
    sign = 1.0 if problem.minimize_objective else -1.0
    bindings = [
        MetricBinding(
            name=spec.name,
            role="objective",
            gemseo_name=objective.name,
            index=index,
            sign=sign,
            minimize=spec.minimize,
            threshold=spec.threshold,
        )
        for index, spec in enumerate(objectives)
    ]
    for function, spec in zip(problem.constraints, constraints, strict=True):
        if list(function.output_names) != [spec.name]:
            raise OptimizationConfigurationError(
                f"the constraint {function.name!r} computes "
                f"{list(function.output_names)}, but the study declares the "
                f"constraint {spec.name!r} at this position"
            )
        bindings.append(
            MetricBinding(
                name=spec.name,
                role="constraint",
                gemseo_name=function.name,
                index=0,
                sign=1.0 if spec.op == "<=" else -1.0,
                offset=gemseo_bound(spec),
                op=spec.op,
                bound=spec.bound,
                tolerance=spec.tolerance,
                scale=spec.scale or 1.0,
            )
        )
    return tuple(bindings)


def add_constraints(scenario: Any, constraints: Sequence[ConstraintSpec]) -> None:
    """Add the constraints of a study to a GEMSEO scenario.

    The tolerance is folded into the bound, so that GEMSEO, which runs with a
    zero tolerance, agrees with the feasibility of the library.

    Args:
        scenario: GEMSEO scenario to add the constraints to.
        constraints: Constraint declarations.
    """
    for spec in constraints:
        scenario.add_constraint(
            spec.name,
            constraint_type="ineq",
            value=gemseo_bound(spec),
            positive=spec.op == ">=",
        )


def _evaluations_done(counter: EvaluationCounter) -> int:
    """Return the new evaluations GEMSEO has started.

    GEMSEO enables the counter on the first new evaluation and increments it
    only when the next one starts, so ``current`` lags by one.
    """
    return counter.current + 1 if counter.enabled else counter.current


class BaseBOLibrary(BaseOptimizationLibrary):
    """GEMSEO driver running an ask-evaluate-tell loop for any backend.

    The driver owns everything that must not depend on the backend: the
    budget, the time limit, the guarded start point, the classification of
    failures, the bindings between user names and GEMSEO functions, and the
    feasibility of a trial. A backend only proposes candidates and learns
    their outcome. It never sees GEMSEO names or NaN, never decides to stop
    and never evaluates x0.

    Attributes:
        result: Outcome of the last run, set even when the run is aborted.
    """

    MAX_STALLED_GENERATIONS = 5
    """Generations adding no new evaluation that end the run."""

    def __init__(self, algo_name: str) -> None:
        super().__init__(algo_name=algo_name)
        self.result: BORunResult | None = None
        self._space: BOSpace | None = None
        self._bindings: tuple[MetricBinding, ...] = ()
        self._start: dict[str, Any] = {}
        self._records: list[TrialRecord] = []

    @abstractmethod
    def _setup(
        self,
        space: BOSpace,
        bindings: tuple[MetricBinding, ...],
        prior: tuple[TrialRecord, ...],
    ) -> None:
        """Prepare the backend before the first ask.

        Args:
            space: Design space of the run.
            bindings: Objectives and constraints, by user name.
            prior: Trials already evaluated: the start point, if any.
        """

    @abstractmethod
    def _ask(self, n: int) -> list[Candidate]:
        """Propose candidates.

        Args:
            n: Maximum number of candidates to propose.

        Returns:
            At most ``n`` candidates, none if the backend has nothing to
            propose. More are tolerated: the driver abandons the surplus.
        """

    @abstractmethod
    def _tell(self, candidate: Candidate, outcome: TrialOutcome) -> None:
        """Report what became of a candidate.

        Args:
            candidate: Candidate returned by ``_ask``.
            outcome: Its outcome. A failed one has no metrics, an abandoned one
                was never evaluated. A candidate already evaluated gets the
                outcome it had.
        """

    def _check_stopping_criteria(self) -> None:
        BaseDriverLibrary._check_stopping_criteria(self)

    def _finalize_previous_iteration(self) -> None:
        # GEMSEO updates its progress bar from the last stored point, and a
        # failed evaluation stores none: an empty database must not abort.
        if len(self._problem.database) == 0:
            self._problem.evaluation_counter.current += 1
            return
        super()._finalize_previous_iteration()

    def _pre_run(self, problem: OptimizationProblem) -> None:
        settings = self._settings
        self.result = None
        self._records = []
        self._bindings = build_metric_bindings(
            problem, settings.objectives, settings.constraints
        )
        self._space = self._build_space()
        self._start = self._space.start_point()
        problem.design_space.set_current_value(self._space.to_vector(self._start))
        problem.stop_if_nan = False
        self._check_constraints_handling(problem)
        self._init_iter_observer(
            problem,
            settings.max_iter,
            message="",
            progress_bar_data_name=settings.progress_bar_data_name,
        )

    def _build_space(self) -> BOSpace:
        settings = self._settings
        try:
            linear_constraints = parse_parameter_constraints(
                settings.parameter_constraints, settings.design_variables
            )
        except ParameterConstraintError as error:
            raise OptimizationConfigurationError(str(error)) from error
        return BOSpace(
            design_variables=settings.design_variables,
            linear_constraints=tuple(linear_constraints),
            parameter_constraints=settings.parameter_constraints,
        )

    def _run(self, problem: OptimizationProblem) -> tuple[Any, Any]:
        stop_reason: StopReason = "aborted"
        try:
            stop_reason = self._drive()
        finally:
            self.result = BORunResult(tuple(self._records), stop_reason, self._bindings)
        return f"Optimization stopped: {stop_reason}.", None

    def _drive(self) -> StopReason:
        settings = self._settings
        counter = self._problem.evaluation_counter
        if settings.x0_enabled:
            stop = self._try(Candidate(self._start), "x0")
            if stop is not None:
                return stop
        self._setup(self._space, self._bindings, tuple(self._records))
        stalled = 0
        while True:
            remaining = settings.max_iter - _evaluations_done(counter)
            if remaining <= 0:
                return "budget"
            candidates = self._ask(min(settings.batch_size, remaining))
            evaluated = _evaluations_done(counter)
            stop: StopReason | None = None
            for candidate in candidates:
                if stop is None:
                    stop = self._try(candidate, self._next_phase())
                else:
                    self._abandon(candidate, stop)
            if stop is not None:
                return stop
            stalled = 0 if _evaluations_done(counter) > evaluated else stalled + 1
            if stalled >= self.MAX_STALLED_GENERATIONS:
                return "search_space_exhausted"

    def _next_phase(self) -> Phase:
        started = sum(1 for record in self._records if record.phase != "x0")
        return "init" if started < self._settings.n_init else "bo"

    def _try(self, candidate: Candidate, phase: Phase) -> StopReason | None:
        """Evaluate a candidate, record it, and tell the backend.

        A point GEMSEO already evaluated is not evaluated again, nor recorded:
        the backend is only told what it got the first time.

        Returns:
            The reason to stop the run, ``None`` to go on.
        """
        counter = self._problem.evaluation_counter
        before = _evaluations_done(counter)
        stop: StopReason | None = None
        try:
            outcome = self._evaluate(candidate)
        except MaxIterReachedException:
            stop = "budget"
        except MaxTimeReached:
            stop = "max_time"
        if stop is not None:
            self._abandon(candidate, stop, phase)
            return stop
        if _evaluations_done(counter) > before:
            self._records.append(
                TrialRecord(len(self._records), phase, candidate.parameters, outcome)
            )
        if phase != "x0":
            self._tell(candidate, outcome)
        if self._failures_in_a_row() >= self._settings.max_consecutive_failures:
            return "consecutive_failures"
        return None

    def _abandon(
        self, candidate: Candidate, reason: StopReason, phase: Phase | None = None
    ) -> None:
        outcome = TrialOutcome("abandoned", reason=reason)
        phase = phase or self._next_phase()
        self._records.append(
            TrialRecord(len(self._records), phase, candidate.parameters, outcome)
        )
        if phase != "x0":
            self._tell(candidate, outcome)

    def _failures_in_a_row(self) -> int:
        count = 0
        for record in reversed(self._records):
            if record.outcome.status != "failed":
                break
            count += 1
        return count

    def _evaluate(self, candidate: Candidate) -> TrialOutcome:
        x = self._space.to_vector(candidate.parameters)
        try:
            output_data, _ = self._problem.evaluate_functions(
                x, design_vector_is_normalized=False
            )
        except EvaluationError as error:
            return TrialOutcome("failed", reason=f"{error.code}: {error}")
        metrics = {
            binding.name: binding.user_value(output_data) for binding in self._bindings
        }
        invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
        if invalid:
            return TrialOutcome(
                "failed", reason=f"NON_FINITE_OUTPUT: {', '.join(invalid)}"
            )
        return TrialOutcome("completed", metrics)
