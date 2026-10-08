"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog

from mdo_framework.core.topology import to_parameter_definition
from mdo_framework.optimization.errors import OptimizationConfigurationError
from mdo_framework.optimization.parameter_codec import (
    ParameterValueError,
    decode_parameter_value,
    encode_parameter_value,
    value_to_index,
)
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    DesignVariable,
    RangeVar,
    Scalar,
)
from mdo_framework.validation import LinearConstraint

StopReason = Literal[
    "budget", "max_time", "search_space_exhausted", "consecutive_failures"
]
TrialStatus = Literal["completed", "failed", "abandoned"]
Phase = Literal["x0", "init", "bo"]

LINEAR_CONSTRAINT_SLACK = 1e-9


@dataclass(frozen=True)
class Candidate:
    """A point proposed by a backend.

    Attributes:
        parameters: User values of the design variables, in schema order.
        key: Backend handle used to report the outcome, e.g. an Ax trial index.
    """

    parameters: Mapping[str, Scalar]
    key: Hashable | None = None


@dataclass(frozen=True)
class TrialOutcome:
    """What evaluating a candidate produced.

    Attributes:
        status: ``"completed"`` if every metric is available, ``"failed"`` if
            the evaluation did not produce usable values, ``"abandoned"`` if
            it was never evaluated because the run stopped.
        metrics: Raw user value of each objective and constraint output.
        reason: Why the trial failed or was abandoned, ``None`` otherwise.
    """

    status: TrialStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None


def gemseo_bound(spec: ConstraintSpec) -> float:
    """Return the bound GEMSEO enforces for a constraint.

    The tolerance is folded into the bound, so GEMSEO runs with a zero
    tolerance and its feasibility matches ``MetricBinding.satisfied``.

    Args:
        spec: Constraint declaration.

    Returns:
        ``bound + tolerance`` for ``<=``, ``bound - tolerance`` for ``>=``.
    """
    if spec.op == "<=":
        return spec.bound + spec.tolerance
    return spec.bound - spec.tolerance


@dataclass(frozen=True)
class MetricBinding:
    """Link between a user output and the GEMSEO function that computes it.

    The user value is ``sign * gemseo_value[index] + offset``.

    Attributes:
        name: User output name of the objective or constraint.
        role: ``"objective"`` or ``"constraint"``.
        gemseo_name: Name of the GEMSEO function holding the value.
        index: Position of the value in that function's output.
        sign: Factor undoing the GEMSEO transformation.
        offset: Term undoing the GEMSEO transformation, in user units.
        minimize: Direction of an objective.
        threshold: Reference point of an objective, in user units.
        op: ``"<="`` or ``">="`` for a constraint.
        bound: Bound of a constraint, in user units.
        tolerance: Violation of the bound still accepted, in user units.
        scale: Typical magnitude of a constraint, to compare violations.
    """

    name: str
    role: Literal["objective", "constraint"]
    gemseo_name: str
    index: int
    sign: float
    offset: float = 0.0
    minimize: bool = True
    threshold: float | None = None
    op: Literal["<=", ">="] | None = None
    bound: float | None = None
    tolerance: float = 0.0
    scale: float = 1.0

    def user_value(self, output_data: Mapping[str, Any]) -> float:
        """Return the raw user value from the output data of GEMSEO.

        Args:
            output_data: GEMSEO function values, by GEMSEO function name.

        Returns:
            The value of the user output, in user units.
        """
        values = np.asarray(output_data[self.gemseo_name], dtype=float).ravel()
        return float(self.sign * values[self.index] + self.offset)

    def margin(self, value: float) -> float:
        """Return the distance to the bound, positive when inside it.

        Args:
            value: Raw user value of the constraint output.

        Returns:
            ``bound - value`` for ``<=``, ``value - bound`` for ``>=``.

        Raises:
            ValueError: If the binding is not a constraint.
        """
        if self.role != "constraint" or self.op is None or self.bound is None:
            raise ValueError(f"{self.name!r} is not a constraint, it has no margin")
        if self.op == "<=":
            return self.bound - value
        return value - self.bound

    def satisfied(self, value: float) -> bool:
        """Return whether a value respects the bound within the tolerance.

        Args:
            value: Raw user value of the output.

        Returns:
            ``margin >= -tolerance`` for a constraint, always ``True`` for an
            objective.
        """
        if self.role != "constraint":
            return True
        return self.margin(value) >= -self.tolerance


@dataclass(frozen=True)
class TrialRecord:
    """A candidate together with its outcome.

    Attributes:
        index: Position of the trial in the run.
        phase: ``"x0"`` for the start point, ``"init"`` for the initial
            design, ``"bo"`` for the model-driven trials.
        parameters: User values of the design variables.
        outcome: What the evaluation produced.
    """

    index: int
    phase: Phase
    parameters: Mapping[str, Scalar]
    outcome: TrialOutcome

    def feasible(self, bindings: Sequence[MetricBinding]) -> bool:
        """Return whether the trial completed within every constraint."""
        if self.outcome.status != "completed":
            return False
        return all(
            binding.satisfied(self.outcome.metrics[binding.name])
            for binding in _constraints(bindings)
        )

    def violation(self, bindings: Sequence[MetricBinding]) -> float:
        """Return the total violation, net of tolerance, in constraint scales."""
        return sum(
            max(
                0.0,
                -(
                    binding.margin(self.outcome.metrics[binding.name])
                    + binding.tolerance
                ),
            )
            / binding.scale
            for binding in _constraints(bindings)
        )

    def to_history_entry(self, bindings: Sequence[MetricBinding]) -> dict[str, Any]:
        """Return the record as a history entry keyed by user names only.

        Args:
            bindings: Objective and constraint bindings of the run.

        Returns:
            The entry. ``objectives`` holds every metric and ``constraints``
            their margins; both are empty unless the trial completed.
        """
        completed = self.outcome.status == "completed"
        metrics = self.outcome.metrics
        return {
            "index": self.index,
            "phase": self.phase,
            "status": self.outcome.status,
            "reason": self.outcome.reason,
            "parameters": dict(self.parameters),
            "objectives": (
                {binding.name: metrics[binding.name] for binding in bindings}
                if completed
                else {}
            ),
            "constraints": (
                {
                    binding.name: {
                        "value": metrics[binding.name],
                        "margin": binding.margin(metrics[binding.name]),
                        "satisfied": binding.satisfied(metrics[binding.name]),
                    }
                    for binding in _constraints(bindings)
                }
                if completed
                else {}
            ),
            "feasible": self.feasible(bindings),
        }


def _constraints(bindings: Sequence[MetricBinding]) -> list[MetricBinding]:
    return [binding for binding in bindings if binding.role == "constraint"]


def _objectives(bindings: Sequence[MetricBinding]) -> list[MetricBinding]:
    return [binding for binding in bindings if binding.role == "objective"]


def _minimised(record: TrialRecord, objectives: Sequence[MetricBinding]) -> list[float]:
    """Objective values of a completed record, signed so lower is better."""
    return [
        record.outcome.metrics[objective.name] * (1.0 if objective.minimize else -1.0)
        for objective in objectives
    ]


def _dominates(better: Sequence[float], other: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(better, other, strict=True)) and any(
        a < b for a, b in zip(better, other, strict=True)
    )


def _pareto_front(
    records: Sequence[TrialRecord], bindings: Sequence[MetricBinding]
) -> list[TrialRecord]:
    objectives = _objectives(bindings)
    feasible = [record for record in records if record.feasible(bindings)]
    points = [_minimised(record, objectives) for record in feasible]
    return [
        record
        for record, point in zip(feasible, points, strict=True)
        if not any(_dominates(other, point) for other in points)
    ]


def _compromise(
    front: Sequence[TrialRecord], objectives: Sequence[MetricBinding]
) -> TrialRecord:
    """Front point closest to the ideal point once each objective is min-max scaled."""
    points = np.array([_minimised(record, objectives) for record in front])
    lowest = points.min(axis=0)
    span = points.max(axis=0) - lowest
    scaled = (points - lowest) / np.where(span > 0.0, span, 1.0)
    return front[int(np.argmin(np.linalg.norm(scaled, axis=1)))]


def select_best(
    records: Sequence[TrialRecord], bindings: Sequence[MetricBinding]
) -> TrialRecord | None:
    """Pick the record to report as the result of a run.

    Args:
        records: Trial records, in run order.
        bindings: Objective and constraint bindings of the run.

    Returns:
        With one objective, the feasible completed record that is best in the
        user direction; with several, the Pareto-front point closest to the
        ideal point. Without a feasible record, the completed record with the
        least total violation. Earlier records win ties. ``None`` if no trial
        completed.
    """
    objectives = _objectives(bindings)
    if any(record.feasible(bindings) for record in records):
        if len(objectives) == 1:
            return min(
                (record for record in records if record.feasible(bindings)),
                key=lambda record: _minimised(record, objectives),
            )
        return _compromise(_pareto_front(records, bindings), objectives)
    completed = [record for record in records if record.outcome.status == "completed"]
    return min(completed, key=lambda record: record.violation(bindings), default=None)


@dataclass(frozen=True)
class BORunResult:
    """Outcome of a Bayesian optimization run.

    Attributes:
        records: Every trial, in run order.
        stop_reason: Why the run stopped.
        bindings: Objective and constraint bindings of the run.
    """

    records: Sequence[TrialRecord]
    stop_reason: StopReason
    bindings: Sequence[MetricBinding]

    @property
    def evaluations(self) -> dict[str, int]:
        """Evaluations per phase, which exclude abandoned trials, and failures."""
        evaluated = [r for r in self.records if r.outcome.status != "abandoned"]
        counts = {
            phase: sum(record.phase == phase for record in evaluated)
            for phase in ("x0", "init", "bo")
        }
        counts["failed"] = sum(r.outcome.status == "failed" for r in self.records)
        return counts

    @property
    def best(self) -> TrialRecord | None:
        """The record to report, see ``select_best``."""
        return select_best(self.records, self.bindings)

    @property
    def pareto_front(self) -> list[TrialRecord]:
        """Feasible records not dominated by another, empty for one objective."""
        if len(_objectives(self.bindings)) < 2:
            return []
        return _pareto_front(self.records, self.bindings)

    @property
    def feasible(self) -> bool:
        """Whether the reported record respects every constraint."""
        best = self.best
        return best is not None and best.feasible(self.bindings)


def _round_half_up(value: float, variable: RangeVar) -> int:
    rounded = math.floor(value + 0.5)
    return int(min(max(rounded, variable.lower), variable.upper))


def _centre(variable: DesignVariable) -> Scalar:
    if isinstance(variable, ChoiceVar):
        return variable.choices[(len(variable.choices) - 1) // 2]
    middle = (variable.lower + variable.upper) / 2
    if variable.value_type == "int":
        return _round_half_up(middle, variable)
    return middle


def _initial_or_centre(variable: DesignVariable) -> Scalar:
    if variable.initial is None:
        return _centre(variable)
    if isinstance(variable, RangeVar) and variable.value_type == "int":
        return int(variable.initial)
    return variable.initial


def _admits(variable: DesignVariable, value: Any) -> bool:
    """Whether a value is within the bounds, integral or a declared choice."""
    if isinstance(variable, ChoiceVar):
        try:
            value_to_index(to_parameter_definition(variable), value)
        except ParameterValueError:
            return False
        return True
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return False
    number = float(value)
    if not variable.lower <= number <= variable.upper:
        return False
    return variable.value_type != "int" or number.is_integer()


def _largest_inscribed_ball(
    ranges: Sequence[RangeVar], constraints: Sequence[LinearConstraint]
) -> np.ndarray:
    """Centre of the largest ball inside the constraints, in unit-box coordinates.

    Working on ``(x - lower) / (upper - lower)`` keeps variables of very
    different magnitudes comparable.
    """
    count = len(ranges)
    position = {variable.name: i for i, variable in enumerate(ranges)}
    lower = np.array([variable.lower for variable in ranges])
    width = np.array([variable.upper - variable.lower for variable in ranges])
    coefficients = np.zeros((len(constraints), count))
    for row, constraint in enumerate(constraints):
        for name, coefficient in constraint.coefficients.items():
            coefficients[row, position[name]] = coefficient
    unit_coefficients = coefficients * width
    ball_rows = np.hstack(
        [unit_coefficients, np.linalg.norm(unit_coefficients, axis=1, keepdims=True)]
    )
    ball_bounds = (
        np.array([constraint.bound for constraint in constraints])
        - coefficients @ lower
    )
    identity, radius = np.eye(count), np.ones((count, 1))
    box_rows = np.vstack(
        [np.hstack([identity, radius]), np.hstack([-identity, radius])]
    )
    box_bounds = np.concatenate([np.ones(count), np.zeros(count)])
    result = linprog(
        c=np.append(np.zeros(count), -1.0),
        A_ub=np.vstack([ball_rows, box_rows]),
        b_ub=np.concatenate([ball_bounds, box_bounds]),
        bounds=[(None, None)] * count + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise OptimizationConfigurationError(
            f"The parameter constraints make the design space empty: {result.message}"
        )
    return result.x[:count]


@dataclass(frozen=True)
class BOSpace:
    """Design space of a run: variables and linear constraints between them.

    Attributes:
        design_variables: Range and choice variables, in schema order.
        linear_constraints: The parameter constraints, parsed.
        parameter_constraints: The same constraints as expressions, for
            backends that take them verbatim.
    """

    design_variables: tuple[DesignVariable, ...]
    linear_constraints: tuple[LinearConstraint, ...] = ()
    parameter_constraints: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Design variable names, in schema order."""
        return tuple(variable.name for variable in self.design_variables)

    def to_vector(self, parameters: Mapping[str, Scalar]) -> np.ndarray:
        """Encode user values as a GEMSEO design vector.

        Args:
            parameters: User value of every design variable.

        Returns:
            The vector, in schema order, with choices as indices.
        """
        return np.array(
            [
                encode_parameter_value(
                    to_parameter_definition(variable), parameters[variable.name]
                )
                for variable in self.design_variables
            ],
            dtype=float,
        )

    def from_vector(self, x: Sequence[float] | np.ndarray) -> dict[str, Scalar]:
        """Decode a GEMSEO design vector into user values.

        Args:
            x: Design vector, in schema order.

        Returns:
            User values: floats, ints, and the declared choice values.
        """
        return {
            variable.name: decode_parameter_value(
                to_parameter_definition(variable), value
            )
            for variable, value in zip(self.design_variables, x, strict=True)
        }

    def _violated_constraint(
        self, parameters: Mapping[str, Scalar]
    ) -> LinearConstraint | None:
        for constraint in self.linear_constraints:
            total = sum(
                coefficient * float(parameters[name])
                for name, coefficient in constraint.coefficients.items()
            )
            limit = constraint.bound + LINEAR_CONSTRAINT_SLACK * max(
                1.0, abs(constraint.bound)
            )
            if not total <= limit:
                return constraint
        return None

    def contains(self, parameters: Mapping[str, Scalar]) -> bool:
        """Return whether a point is in the design space.

        Args:
            parameters: User value of every design variable.

        Returns:
            ``True`` if every value is within bounds, integral for integer
            variables and a declared choice, and the linear constraints hold.
        """
        for variable in self.design_variables:
            if variable.name not in parameters:
                return False
            if not _admits(variable, parameters[variable.name]):
                return False
        return self._violated_constraint(parameters) is None

    def box_centre(self) -> dict[str, Scalar]:
        """Return the centre of the variable bounds, ignoring constraints."""
        return {variable.name: _centre(variable) for variable in self.design_variables}

    def chebyshev_centre(self) -> dict[str, Scalar]:
        """Return the centre of the largest ball inside the linear constraints.

        The ball is measured in unit-box coordinates over the range variables;
        integers are rounded and choices sit at the box centre.

        Raises:
            OptimizationConfigurationError: If the constraints leave no point
                in the design space, or rounding integers leaves it.
        """
        point = self.box_centre()
        if not self.linear_constraints:
            return point
        ranges = [v for v in self.design_variables if isinstance(v, RangeVar)]
        centres = _largest_inscribed_ball(ranges, self.linear_constraints)
        for variable, unit in zip(ranges, centres, strict=True):
            value = variable.lower + unit * (variable.upper - variable.lower)
            if variable.value_type == "int":
                point[variable.name] = _round_half_up(value, variable)
            else:
                point[variable.name] = float(value)
        violated = self._violated_constraint(point)
        if violated is not None:
            raise OptimizationConfigurationError(
                "The Chebyshev centre rounded to integers violates the parameter "
                f"constraint {violated.expression!r}."
            )
        return point

    def start_point(self) -> dict[str, Scalar]:
        """Return the point the run starts from.

        Declared ``initial`` values are used, the box centre elsewhere. If that
        violates a linear constraint and no variable declares an initial value,
        the Chebyshev centre is used instead.

        Raises:
            OptimizationConfigurationError: If declared initial values violate
                a linear constraint, or no Chebyshev centre exists.
        """
        point = {
            variable.name: _initial_or_centre(variable)
            for variable in self.design_variables
        }
        violated = self._violated_constraint(point)
        if violated is None:
            return point
        if all(variable.initial is None for variable in self.design_variables):
            return self.chebyshev_centre()
        raise OptimizationConfigurationError(
            f"The declared initial values violate the parameter constraint "
            f"{violated.expression!r}; change them or remove them to let the "
            "optimizer choose a feasible start point."
        )
