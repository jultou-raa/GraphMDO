"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from ax.adapter.registry import Generators as Models
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError, OptimizationComplete
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription

from mdo_framework.optimization.bo_library import BaseBOLibrary, BaseBOSettings
from mdo_framework.optimization.bo_types import (
    BOSpace,
    Candidate,
    MetricBinding,
    TrialOutcome,
    TrialRecord,
)
from mdo_framework.optimization.errors import OptimizationExecutionError
from mdo_framework.schema import ChoiceVar, DesignVariable, RangeVar

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "GraphMDO"


@contextmanager
def _quiet_ax() -> Iterator[None]:
    """Hide the FutureWarnings Ax raises from its own modules."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"ax\.")
        yield


def _parameter_config(
    variable: DesignVariable,
) -> RangeParameterConfig | ChoiceParameterConfig:
    """Return the Ax parameter of a design variable, in user values."""
    if isinstance(variable, ChoiceVar):
        ordered = variable.ordered
        if ordered is None:
            # Ax default, made explicit to avoid a warning for each choice parameter
            ordered = variable.value_type != "str" or len(variable.choices) == 2
        return ChoiceParameterConfig(
            name=variable.name,
            values=list(variable.choices),
            parameter_type=variable.value_type,
            is_ordered=ordered,
        )
    return _range_config(variable)


def _range_config(variable: RangeVar) -> RangeParameterConfig:
    if variable.value_type == "int":
        bounds: tuple[float, float] = (
            math.ceil(variable.lower),
            math.floor(variable.upper),
        )
    else:
        bounds = (float(variable.lower), float(variable.upper))
    return RangeParameterConfig(
        name=variable.name,
        bounds=bounds,
        parameter_type=variable.value_type,
        scaling="log" if variable.scaling == "log" else None,
    )


def _objective(binding: MetricBinding) -> Objective:
    return Objective(metric=MapMetric(name=binding.name), minimize=binding.minimize)


def _outcome_constraint(binding: MetricBinding) -> OutcomeConstraint:
    """Return the constraint on the metric ``name`` scaled to its magnitude."""
    op = ComparisonOp.LEQ if binding.op == "<=" else ComparisonOp.GEQ
    return OutcomeConstraint(
        metric=MapMetric(name=binding.name),
        op=op,  # type: ignore[arg-type]  # Pyright widens IntEnum to int
        bound=binding.bound / binding.scale,
        relative=False,
    )


def _objective_threshold(binding: MetricBinding) -> ObjectiveThreshold:
    op = ComparisonOp.LEQ if binding.minimize else ComparisonOp.GEQ
    return ObjectiveThreshold(
        metric=MapMetric(name=binding.name),
        bound=float(binding.threshold),
        relative=False,
        op=op,  # type: ignore[arg-type]  # Pyright widens IntEnum to int
    )


def _optimization_config(
    bindings: tuple[MetricBinding, ...],
) -> OptimizationConfig | MultiObjectiveOptimizationConfig:
    """Return the Ax optimization config, in the user names of the outputs."""
    objectives = [b for b in bindings if b.role == "objective"]
    constraints = [_outcome_constraint(b) for b in bindings if b.role == "constraint"]
    if len(objectives) == 1:
        return OptimizationConfig(
            objective=_objective(objectives[0]), outcome_constraints=constraints
        )
    thresholds = [
        _objective_threshold(b) for b in objectives if b.threshold is not None
    ]
    return MultiObjectiveOptimizationConfig(
        objective=MultiObjective(objectives=[_objective(b) for b in objectives]),
        objective_thresholds=thresholds or None,
        outcome_constraints=constraints,
    )


def _raw_data(
    bindings: tuple[MetricBinding, ...], metrics: dict[str, float]
) -> dict[str, float]:
    """Return the metrics Ax models: constraints are divided by their scale."""
    return {
        b.name: metrics[b.name] / b.scale if b.role == "constraint" else metrics[b.name]
        for b in bindings
    }


class AxSettings(BaseBOSettings):
    """Settings of the Ax backend.

    Attributes:
        use_bonsai: Whether to use the experimental BONSAI acquisition of Ax
            instead of the default one.
    """

    use_bonsai: bool = False


class AxOptimizationLibrary(BaseBOLibrary):
    """Bayesian optimization backend running on the Ax client API.

    It only maps the three hooks of the driver onto an Ax experiment: the
    search space, the metrics and the generation strategy of the experiment,
    the trials Ax proposes, and the outcome of each. Everything else belongs to
    the driver.
    """

    LIBRARY_NAME = "Ax_Platform"

    ALGORITHM_INFOS = {
        "Ax_Bayesian": OptimizationAlgorithmDescription(
            algorithm_name="Ax_Bayesian",
            internal_algorithm_name="Ax_Bayesian",
            library_name=LIBRARY_NAME,
            description="Bayesian optimization with the Ax platform.",
            website="https://ax.dev/",
            Settings=AxSettings,
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            handle_multiobjective=True,
            handle_integer_variables=True,
            positive_constraints=False,
            require_gradient=False,
            for_linear_problems=False,
        )
    }

    def __init__(
        self,
        algo_name: str = "Ax_Bayesian",
        client_factory: Callable[..., Client] | None = None,
    ) -> None:
        """Create the library.

        Args:
            algo_name: Name of the algorithm.
            client_factory: Callable building the Ax client from the keyword
                ``random_seed``. By default, the ``Client`` of this module,
                looked up when a run starts.
        """
        super().__init__(algo_name=algo_name)
        self._client_factory = client_factory
        self._client: Client | None = None
        self._failed_points: set[tuple[Any, ...]] = set()

    def _setup(
        self,
        space: BOSpace,
        bindings: tuple[MetricBinding, ...],
        prior: tuple[TrialRecord, ...],
    ) -> None:
        settings = self._settings
        factory = self._client_factory or Client
        self._failed_points = set()
        with _quiet_ax():
            client = factory(random_seed=settings.seed)
            client.configure_experiment(
                parameters=[_parameter_config(v) for v in space.design_variables],
                parameter_constraints=list(space.parameter_constraints) or None,
                name=EXPERIMENT_NAME,
            )
            client.set_optimization_config(_optimization_config(bindings))
            client.set_generation_strategy(self._generation_strategy(bindings))
            self._client = client
            for record in prior:
                self._attach(record)

    def _generation_strategy(
        self, bindings: tuple[MetricBinding, ...]
    ) -> GenerationStrategy:
        """Return Sobol for the initial trials, then BoTorch.

        The acquisition function is hard-coded for a single objective only: a
        multi-objective run needs the hypervolume one Ax picks itself. Both
        steps reject points of earlier trials, abandoned ones included.
        """
        settings = self._settings
        n_objectives = sum(1 for b in bindings if b.role == "objective")
        generator_kwargs = None
        if settings.use_bonsai:
            logger.warning("Experimental feature BONSAI algorithm is activated.")
        elif n_objectives == 1:
            generator_kwargs = {"botorch_acqf_class": qLogNoisyExpectedImprovement}
        return GenerationStrategy(
            name="bonsai" if settings.use_bonsai else "botorch_modular",
            nodes=[
                GenerationStep(
                    generator=Models.SOBOL,
                    num_trials=settings.n_init,
                    min_trials_observed=1,
                    should_deduplicate=True,
                ),
                GenerationStep(
                    generator=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    generator_kwargs=generator_kwargs,
                    should_deduplicate=True,
                ),
            ],
        )

    def _attach(self, record: TrialRecord) -> None:
        """Attach a point evaluated before the run as the baseline trial."""
        index = self._client.attach_baseline(parameters=dict(record.parameters))
        self._report(index, record.parameters, record.outcome)

    def _ask(self, n: int) -> list[Candidate]:
        with _quiet_ax():
            try:
                trials = self._client.get_next_trials(max_trials=n)
            except DataRequiredError as error:
                raise OptimizationExecutionError(
                    "Ax cannot propose a new design before the initial trials "
                    f"have an outcome: {error}"
                ) from error
            except OptimizationComplete:
                # Ax found no new point; the driver decides when to stop.
                return []
        return [Candidate(dict(p), key=index) for index, p in trials.items()]

    def _tell(self, candidate: Candidate, outcome: TrialOutcome) -> None:
        with _quiet_ax():
            self._report(candidate.key, candidate.parameters, outcome)

    def _report(
        self, index: int, parameters: Mapping[str, Any], outcome: TrialOutcome
    ) -> None:
        if outcome.status == "completed":
            self._client.complete_trial(
                trial_index=index, raw_data=_raw_data(self._bindings, outcome.metrics)
            )
            return
        point = tuple(parameters[name] for name in self._space.names)
        if outcome.status == "failed" and point not in self._failed_points:
            self._failed_points.add(point)
            self._client.mark_trial_failed(
                trial_index=index, failed_reason=outcome.reason
            )
            return
        # Ax retries the points of failed trials but avoids abandoned ones. The
        # driver never evaluates a failed point twice, so a repeat is abandoned.
        self._client.mark_trial_abandoned(trial_index=index)
