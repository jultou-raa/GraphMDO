"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import math
from collections.abc import Mapping

import numpy as np
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription

from mdo_framework.optimization.bo_library import BaseBOLibrary, BaseBOSettings
from mdo_framework.optimization.bo_types import (
    BOSpace,
    Candidate,
    MetricBinding,
    TrialOutcome,
    TrialRecord,
)
from mdo_framework.optimization.errors import OptimizationConfigurationError
from mdo_framework.schema import ChoiceVar, DesignVariable, RangeVar, Scalar


class RandomSearchSettings(BaseBOSettings):
    """Settings of the random-search backend, those of the driver."""


class RandomSearchLibrary(BaseBOLibrary):
    """Seeded random search, the reference backend of the driver.

    It draws every design variable uniformly (log-uniformly on a log scale,
    among the choices for a choice variable), keeps the draws that satisfy the
    parameter constraints, and never proposes a point twice. It learns nothing
    from the outcomes, so it needs no objective model.
    """

    LIBRARY_NAME = "GraphMDO"

    ALGORITHM_INFOS = {
        "BO_RandomSearch": OptimizationAlgorithmDescription(
            algorithm_name="BO_RandomSearch",
            internal_algorithm_name="BO_RandomSearch",
            library_name=LIBRARY_NAME,
            description="Seeded random search, reference backend of the BO driver.",
            website="",
            Settings=RandomSearchSettings,
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            handle_multiobjective=True,
            handle_integer_variables=True,
            positive_constraints=False,
            require_gradient=False,
            for_linear_problems=False,
        )
    }

    MAX_DRAWS = 1000
    """Draws made to find one new point satisfying the parameter constraints."""

    def __init__(self, algo_name: str = "BO_RandomSearch") -> None:
        super().__init__(algo_name=algo_name)
        self._rng = np.random.default_rng()
        self._seen: set[tuple[Scalar, ...]] = set()

    def _setup(
        self,
        space: BOSpace,
        bindings: tuple[MetricBinding, ...],
        prior: tuple[TrialRecord, ...],
    ) -> None:
        self._rng = np.random.default_rng(self._settings.seed)
        self._seen = {self._key(record.parameters) for record in prior}

    def _ask(self, n: int) -> list[Candidate]:
        candidates = []
        for _ in range(n):
            point = self._draw_new_point()
            if point is None:
                break
            candidates.append(Candidate(point))
        return candidates

    def _tell(self, candidate: Candidate, outcome: TrialOutcome) -> None:
        """Learn nothing: the draws do not depend on the outcomes."""

    def _key(self, parameters: Mapping[str, Scalar]) -> tuple[Scalar, ...]:
        return tuple(parameters[name] for name in self._space.names)

    def _draw_new_point(self) -> dict[str, Scalar] | None:
        """Return a new point satisfying the parameter constraints.

        Raises:
            OptimizationConfigurationError: If no draw satisfies them.
        """
        satisfied = False
        for _ in range(self.MAX_DRAWS):
            point = {
                variable.name: self._draw(variable)
                for variable in self._space.design_variables
            }
            if not self._space.contains(point):
                continue
            satisfied = True
            key = self._key(point)
            if key not in self._seen:
                self._seen.add(key)
                return point
        if not satisfied:
            raise OptimizationConfigurationError(
                f"no point satisfying the parameter constraints "
                f"{list(self._space.parameter_constraints)} found in "
                f"{self.MAX_DRAWS} draws: the constraints leave a space too "
                "small to sample"
            )
        return None

    def _draw(self, variable: DesignVariable) -> Scalar:
        if isinstance(variable, ChoiceVar):
            return variable.choices[int(self._rng.integers(len(variable.choices)))]
        return self._draw_range(variable)

    def _draw_range(self, variable: RangeVar) -> Scalar:
        lower, upper = variable.lower, variable.upper
        if variable.value_type == "int":
            return int(self._rng.integers(math.ceil(lower), math.floor(upper) + 1))
        if variable.scaling == "log":
            value = math.exp(self._rng.uniform(math.log(lower), math.log(upper)))
            return min(max(value, lower), upper)
        return float(self._rng.uniform(lower, upper))
