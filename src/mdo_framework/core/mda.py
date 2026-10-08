"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Sequence
from typing import Any, Final, Literal, Self

from gemseo.core.discipline import Discipline
from gemseo.core.discipline.base_discipline import CacheType
from gemseo.mda.factory import MDAFactory
from gemseo.mda.mda_chain import MDAChain
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from mdo_framework.core.errors import MDANotConvergedError

PARALLEL_INNER_MDA: Final = "MDAJacobi"


class MDASettings(BaseModel):
    """How the strongly coupled tools are solved.

    The default runs the coupled tools one after the other (Gauss-Seidel).
    Running them in parallel requires ``inner_mda_name="MDAJacobi"`` and tools
    declared ``thread_safe``.

    Attributes:
        inner_mda_name: GEMSEO algorithm that solves each coupled group.
        tolerance: Normalized residual at which the coupled group has converged.
        max_mda_iter: Maximum number of iterations of the algorithm.
        max_consecutive_unsuccessful_iterations: Number of consecutive
            iterations without residual decrease after which the algorithm
            stops.
        n_processes: Number of threads running the coupled tools at the same
            time. Above 1 only for ``MDAJacobi``.
        accept_tolerance: Largest normalized residual of a stopped algorithm
            that is still accepted. ``None`` accepts only ``tolerance``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inner_mda_name: Literal["MDAGaussSeidel", "MDAJacobi", "MDANewtonRaphson"] = (
        "MDAGaussSeidel"
    )
    tolerance: PositiveFloat = 1e-6
    max_mda_iter: PositiveInt = 20
    max_consecutive_unsuccessful_iterations: PositiveInt = 8
    n_processes: PositiveInt = 1
    accept_tolerance: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.n_processes > 1 and self.inner_mda_name != PARALLEL_INNER_MDA:
            raise ValueError(
                f"n_processes={self.n_processes} requires "
                f"inner_mda_name='{PARALLEL_INNER_MDA}', not "
                f"'{self.inner_mda_name}'"
            )
        if self.accept_tolerance is not None and self.accept_tolerance < self.tolerance:
            raise ValueError(
                f"accept_tolerance ({self.accept_tolerance:g}) cannot be below "
                f"tolerance ({self.tolerance:g})"
            )
        return self

    @property
    def residual_limit(self) -> float:
        """Largest normalized residual of an accepted coupled group."""
        if self.accept_tolerance is None:
            return self.tolerance
        return self.accept_tolerance


def _inner_mda_settings(settings: MDASettings) -> dict[str, Any]:
    values: dict[str, Any] = {
        "tolerance": settings.tolerance,
        "max_mda_iter": settings.max_mda_iter,
        "max_consecutive_unsuccessful_iterations": (
            settings.max_consecutive_unsuccessful_iterations
        ),
    }
    inner_class = MDAFactory().get_class(settings.inner_mda_name)
    if "n_processes" in inner_class.Settings.model_fields:
        # Without it a parallel algorithm starts one thread per CPU.
        values["n_processes"] = settings.n_processes
    return values


class StrictMDAChain(MDAChain):
    """MDAChain that raises instead of returning unconverged states.

    GEMSEO only logs a warning when an algorithm stops above its tolerance and
    returns the last iterate. Every coupled group of this chain is checked
    once it has run, so each evaluation of the discipline either converged or
    raised ``MDANotConvergedError``.
    """

    def __init__(self, disciplines: Sequence[Discipline], settings: MDASettings):
        """Build the chain.

        Args:
            disciplines: The disciplines to chain; the coupled ones are solved
                together by the algorithm of ``settings``.
            settings: How the coupled groups are solved and accepted.
        """
        super().__init__(
            disciplines,
            inner_mda_name=settings.inner_mda_name,
            inner_mda_settings=_inner_mda_settings(settings),
        )
        self._residual_limit = settings.residual_limit

    def _solve(self) -> None:
        super()._solve()
        for mda in self.inner_mdas:
            residual = mda.io.data.get(self.NORMALIZED_RESIDUAL_NORM)
            if residual is None:
                continue
            last = float(residual[-1])
            # A NaN residual compares false, so it is rejected too.
            if not last <= self._residual_limit:
                couplings = ", ".join(sorted(mda.coupling_structure.strong_couplings))
                raise MDANotConvergedError(
                    f"MDA did not converge: residual {last:.1e} > "
                    f"{self._residual_limit:.1e} (couplings: {couplings})"
                )

    def disable_caches(self) -> None:
        """Stop the chain, its algorithms and nested processes from caching.

        A cached process replays its outputs for inputs it has already seen,
        so a stochastic tool inside it would be called only once. The caches
        of the tools themselves are left to the tools.
        """
        self.set_cache(CacheType.NONE)
        _disable_process_caches(self.mdo_chain)


def _disable_process_caches(discipline: Discipline) -> None:
    children = getattr(discipline, "disciplines", ())
    if not children:
        return
    discipline.set_cache(CacheType.NONE)
    for child in children:
        _disable_process_caches(child)


def build_mda(
    disciplines: Sequence[Discipline], settings: MDASettings | None = None
) -> StrictMDAChain:
    """Chain the disciplines, solving the coupled ones as the settings say.

    Args:
        disciplines: The disciplines of the study.
        settings: How to solve and accept the coupled groups; the sequential
            Gauss-Seidel defaults when omitted.

    Returns:
        A discipline whose evaluation raises ``MDANotConvergedError`` if a
        coupled group stops above its accepted residual.
    """
    return StrictMDAChain(disciplines, settings or MDASettings())
