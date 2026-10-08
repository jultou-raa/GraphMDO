"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import re
import subprocess
import sys

import numpy as np
import pytest

from mdo_framework.optimization.bo_types import (
    BORunResult,
    BOSpace,
    Candidate,
    MetricBinding,
    TrialOutcome,
    TrialRecord,
    gemseo_bound,
    select_best,
)
from mdo_framework.optimization.errors import OptimizationConfigurationError
from mdo_framework.schema import ChoiceVar, ConstraintSpec, RangeVar
from mdo_framework.validation import parse_parameter_constraints


def _objective(name="f", *, minimize=True, gemseo_name=None, index=0, sign=1.0):
    return MetricBinding(
        name=name,
        role="objective",
        gemseo_name=gemseo_name or name,
        index=index,
        sign=sign,
        minimize=minimize,
    )


def _constraint(name="c", *, op="<=", bound=1.0, tolerance=0.0, scale=1.0):
    spec = ConstraintSpec(name=name, bound=bound, op=op, tolerance=tolerance)
    return MetricBinding(
        name=name,
        role="constraint",
        gemseo_name=f"[{name}-{gemseo_bound(spec)}]",
        index=0,
        sign=1.0 if op == "<=" else -1.0,
        offset=gemseo_bound(spec),
        op=op,
        bound=bound,
        tolerance=tolerance,
        scale=scale,
    )


def _space(*variables, constraints=()):
    parsed = parse_parameter_constraints(list(constraints), list(variables))
    return BOSpace(tuple(variables), tuple(parsed), tuple(constraints))


X = RangeVar(name="x", lower=0.0, upper=1.0)
Y = RangeVar(name="y", lower=0.0, upper=1.0)
N = RangeVar(name="n", lower=1, upper=5, value_type="int")
MODE = ChoiceVar(name="mode", choices=["a", "b", "c"])
FLAG = ChoiceVar(name="flag", choices=[True, False])


def test_module_does_not_import_gemseo_or_ax():
    code = (
        "import sys, mdo_framework.optimization.bo_types;"
        "print([m for m in sys.modules if m.split('.')[0] in ('gemseo', 'ax')])"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


class TestGemseoBound:
    def test_tolerance_loosens_the_bound_in_the_feasible_direction(self):
        upper = ConstraintSpec(name="c", bound=2.0, op="<=", tolerance=0.5)
        lower = ConstraintSpec(name="c", bound=2.0, op=">=", tolerance=0.5)

        assert gemseo_bound(upper) == 2.5
        assert gemseo_bound(lower) == 1.5

    def test_no_tolerance_keeps_the_bound(self):
        assert gemseo_bound(ConstraintSpec(name="c", bound=3.0)) == 3.0


class TestMetricBindingUserValue:
    def test_minimised_objective_is_read_as_is(self):
        binding = _objective("f1")

        assert binding.user_value({"f1": np.array([0.25])}) == 0.25

    def test_maximised_objective_undoes_the_gemseo_negation(self):
        binding = _objective("f1", minimize=False, gemseo_name="-f1", sign=-1.0)

        assert binding.user_value({"-f1": np.array([-0.25])}) == 0.25

    def test_multi_objective_reads_its_index_of_the_vector_function(self):
        data = {"f1_f2": np.array([1.5, -2.5])}

        first = _objective("f1", gemseo_name="f1_f2", index=0)
        second = _objective("f2", gemseo_name="f1_f2", index=1)

        assert first.user_value(data) == 1.5
        assert second.user_value(data) == -2.5

    def test_upper_bound_constraint_folds_in_bound_and_tolerance(self):
        binding = _constraint(op="<=", bound=2.0, tolerance=0.5)

        # GEMSEO runs "[c-2.5]" = c - 2.5; c = 2.2 gives -0.3.
        assert binding.user_value({"[c-2.5]": np.array([-0.3])}) == pytest.approx(2.2)

    def test_lower_bound_constraint_folds_in_bound_and_tolerance(self):
        binding = _constraint(op=">=", bound=2.0, tolerance=0.5)

        # GEMSEO runs "-[c-1.5]" = 1.5 - c; c = 1.8 gives -0.3.
        assert binding.user_value({"[c-1.5]": np.array([-0.3])}) == pytest.approx(1.8)

    def test_returns_a_python_float(self):
        value = _objective("f1").user_value({"f1": np.array([2])})

        assert type(value) is float


class TestMetricBindingMargin:
    def test_upper_bound_margin_is_bound_minus_value(self):
        binding = _constraint(op="<=", bound=2.0, tolerance=0.5)

        assert binding.margin(1.5) == 0.5
        assert binding.margin(2.75) == -0.75

    def test_lower_bound_margin_is_value_minus_bound(self):
        binding = _constraint(op=">=", bound=2.0, tolerance=0.25)

        assert binding.margin(2.5) == 0.5
        assert binding.margin(1.75) == -0.25

    def test_upper_bound_tolerance_edges(self):
        binding = _constraint(op="<=", bound=2.0, tolerance=0.5)

        assert binding.satisfied(2.0)
        assert binding.satisfied(2.5)
        assert not binding.satisfied(2.5000001)

    def test_lower_bound_tolerance_edges(self):
        binding = _constraint(op=">=", bound=2.0, tolerance=0.25)

        assert binding.satisfied(2.0)
        assert binding.satisfied(1.75)
        assert not binding.satisfied(1.7499999)

    def test_not_a_number_never_satisfies(self):
        assert not _constraint().satisfied(float("nan"))

    def test_objectives_are_always_satisfied_and_have_no_margin(self):
        binding = _objective()

        assert binding.satisfied(1e30)
        with pytest.raises(ValueError, match="constraint"):
            binding.margin(1.0)


class TestBOSpaceVectors:
    def test_names_keep_the_declaration_order(self):
        assert _space(N, MODE, X).names == ("n", "mode", "x")

    def test_round_trip_float_int_and_choices(self):
        space = _space(X, N, MODE, FLAG)
        parameters = {"x": 0.25, "n": 3, "mode": "c", "flag": False}

        vector = space.to_vector(parameters)
        decoded = space.from_vector(vector)

        assert vector.tolist() == [0.25, 3.0, 2.0, 1.0]
        assert decoded == parameters
        assert type(decoded["n"]) is int
        assert decoded["flag"] is False
        assert list(decoded) == ["x", "n", "mode", "flag"]

    def test_to_vector_rejects_an_undeclared_choice(self):
        with pytest.raises(ValueError, match="not a declared choice"):
            _space(MODE).to_vector({"mode": "z"})


class TestBOSpaceContains:
    def test_accepts_a_point_in_the_space(self):
        space = _space(X, N, MODE)

        assert space.contains({"x": 1.0, "n": 5, "mode": "b"})

    @pytest.mark.parametrize(
        "parameters",
        [
            {"x": 1.5, "n": 3, "mode": "a"},
            {"x": -0.1, "n": 3, "mode": "a"},
            {"x": 0.5, "n": 2.5, "mode": "a"},
            {"x": 0.5, "n": 6, "mode": "a"},
            {"x": 0.5, "n": 3, "mode": "z"},
            {"x": 0.5, "n": 3},
            {"x": float("nan"), "n": 3, "mode": "a"},
            {"x": "0.5", "n": 3, "mode": "a"},
        ],
    )
    def test_rejects_a_point_outside_the_space(self, parameters):
        assert not _space(X, N, MODE).contains(parameters)

    def test_integral_float_is_accepted_for_an_integer_variable(self):
        assert _space(N).contains({"n": 3.0})

    def test_choice_membership_distinguishes_bool_from_int(self):
        assert not _space(FLAG).contains({"flag": 1})
        assert _space(FLAG).contains({"flag": True})

    def test_linear_constraints_are_enforced_with_a_tiny_slack(self):
        space = _space(X, Y, constraints=["x + y <= 0.5"])

        assert space.contains({"x": 0.25, "y": 0.25})
        assert space.contains({"x": 0.25, "y": 0.25 + 1e-12})
        assert not space.contains({"x": 0.4, "y": 0.2})


class TestBOSpaceCentres:
    def test_box_centre_of_range_and_choice_variables(self):
        space = _space(
            X,
            RangeVar(name="k", lower=0, upper=5, value_type="int"),
            RangeVar(name="m", lower=0, upper=4, value_type="int"),
            MODE,
            ChoiceVar(name="four", choices=[1, 2, 3, 4]),
            ChoiceVar(name="two", choices=["p", "q"]),
        )

        centre = space.box_centre()

        assert centre == {"x": 0.5, "k": 3, "m": 2, "mode": "b", "four": 2, "two": "p"}
        assert type(centre["k"]) is int

    def test_chebyshev_centre_is_strictly_inside_the_constraints(self):
        space = _space(X, Y, constraints=["x + y <= 0.5"])

        centre = space.chebyshev_centre()

        assert space.contains(centre)
        assert 0.0 < centre["x"] and 0.0 < centre["y"]
        assert centre["x"] + centre["y"] < 0.5
        assert centre["x"] == pytest.approx(centre["y"])
        assert centre["x"] == pytest.approx((1 - 1 / np.sqrt(2)) / 2)

    def test_chebyshev_centre_is_in_normalised_coordinates(self):
        wide = RangeVar(name="wide", lower=0.0, upper=1000.0)
        space = _space(X, wide, constraints=["x + 0.001*wide <= 0.5"])

        centre = space.chebyshev_centre()

        assert centre["x"] == pytest.approx(centre["wide"] / 1000)

    def test_chebyshev_centre_keeps_integers_and_choices(self):
        n10 = RangeVar(name="n", lower=0, upper=10, value_type="int")
        space = _space(X, n10, MODE, constraints=["x + n <= 5.5"])

        centre = space.chebyshev_centre()

        assert space.contains(centre)
        assert centre["mode"] == "b"
        assert type(centre["n"]) is int
        assert list(centre) == ["x", "n", "mode"]

    def test_chebyshev_centre_without_constraints_is_the_box_centre(self):
        assert _space(X, Y).chebyshev_centre() == {"x": 0.5, "y": 0.5}

    def test_empty_constrained_space_is_a_configuration_error(self):
        space = _space(X, Y, constraints=["x + y <= -1"])

        with pytest.raises(OptimizationConfigurationError, match="empty"):
            space.chebyshev_centre()

    def test_rounding_out_of_the_constraints_is_a_configuration_error(self):
        flag = RangeVar(name="k", lower=0, upper=1, value_type="int")
        space = _space(flag, constraints=["k >= 0.4", "k <= 0.6"])

        with pytest.raises(OptimizationConfigurationError, match="round"):
            space.chebyshev_centre()


class TestBOSpaceStartPoint:
    def test_declared_initials_win_over_the_box_centre(self):
        space = _space(
            RangeVar(name="x", lower=0.0, upper=1.0, initial=0.2),
            RangeVar(name="n", lower=1, upper=5, value_type="int", initial=4),
            ChoiceVar(name="mode", choices=["a", "b"], initial="b"),
        )

        start = space.start_point()

        assert start == {"x": 0.2, "n": 4, "mode": "b"}
        assert type(start["n"]) is int

    def test_box_centre_where_no_initial_is_declared(self):
        space = _space(RangeVar(name="x", lower=0.0, upper=1.0, initial=0.2), N)

        assert space.start_point() == {"x": 0.2, "n": 3}

    def test_chebyshev_centre_when_the_box_centre_violates_and_no_initial(self):
        space = _space(X, Y, constraints=["x + y <= 0.5"])

        start = space.start_point()

        assert start == space.chebyshev_centre()
        assert space.contains(start)

    def test_violating_declared_initials_name_the_constraint(self):
        space = _space(
            RangeVar(name="x", lower=0.0, upper=1.0, initial=0.5),
            RangeVar(name="y", lower=0.0, upper=1.0, initial=0.5),
            constraints=["x + y <= 0.5"],
        )

        with pytest.raises(OptimizationConfigurationError) as raised:
            space.start_point()

        assert re.search(re.escape("x + y <= 0.5"), str(raised.value))
        assert "initial" in str(raised.value)

    def test_a_partial_initial_that_violates_is_not_repaired(self):
        space = _space(
            RangeVar(name="x", lower=0.0, upper=1.0, initial=0.5),
            Y,
            constraints=["x + y <= 0.5"],
        )

        with pytest.raises(OptimizationConfigurationError, match=r"x \+ y <= 0\.5"):
            space.start_point()

    def test_feasible_start_is_returned_untouched(self):
        space = _space(
            RangeVar(name="x", lower=0.0, upper=1.0, initial=0.1),
            RangeVar(name="y", lower=0.0, upper=1.0, initial=0.1),
            constraints=["x + y <= 0.5"],
        )

        assert space.start_point() == {"x": 0.1, "y": 0.1}


class TestTrialRecord:
    BINDINGS = (_objective("f"), _constraint("c", bound=1.0, tolerance=0.25, scale=2.0))

    def _record(self, outcome, *, index=2, phase="bo"):
        return TrialRecord(index, phase, {"x": 0.5}, outcome)

    def test_history_entry_of_a_completed_record_uses_user_names_only(self):
        record = self._record(
            TrialOutcome("completed", {"f": 3.0, "c": 1.125}), phase="init"
        )

        entry = record.to_history_entry(self.BINDINGS)

        assert entry == {
            "index": 2,
            "phase": "init",
            "status": "completed",
            "reason": None,
            "parameters": {"x": 0.5},
            "objectives": {"f": 3.0, "c": 1.125},
            "constraints": {"c": {"value": 1.125, "margin": -0.125, "satisfied": True}},
            "feasible": True,
        }

    def test_failed_record_has_no_metrics_and_keeps_its_reason(self):
        record = self._record(TrialOutcome("failed", reason="TOOL_FAILED: boom"))

        entry = record.to_history_entry(self.BINDINGS)

        assert entry["status"] == "failed"
        assert entry["reason"] == "TOOL_FAILED: boom"
        assert entry["objectives"] == {}
        assert entry["constraints"] == {}
        assert entry["feasible"] is False

    def test_history_entry_ignores_metrics_that_are_not_user_names(self):
        outcome = TrialOutcome("completed", {"f": 1.0, "c": 0.0, "-f": -1.0})

        entry = self._record(outcome).to_history_entry(self.BINDINGS)

        assert set(entry["objectives"]) == {"f", "c"}

    def test_feasible_needs_a_completed_trial_within_every_constraint(self):
        within = TrialOutcome("completed", {"f": 1.0, "c": 1.25})
        violating = TrialOutcome("completed", {"f": 1.0, "c": 1.5})

        assert self._record(within).feasible(self.BINDINGS)
        assert not self._record(violating).feasible(self.BINDINGS)
        assert not self._record(TrialOutcome("failed")).feasible(self.BINDINGS)
        assert not self._record(TrialOutcome("abandoned")).feasible(self.BINDINGS)

    def test_feasible_without_constraints(self):
        record = self._record(TrialOutcome("completed", {"f": 1.0}))

        assert record.feasible((_objective("f"),))


def _record(index, phase="bo", *, status="completed", reason=None, **metrics):
    return TrialRecord(
        index, phase, {"x": float(index)}, TrialOutcome(status, metrics, reason)
    )


class TestRunResultEvaluations:
    def test_counts_by_phase_and_failures_and_ignores_abandoned(self):
        result = BORunResult(
            records=[
                _record(0, "x0", f=1.0),
                _record(1, "init", f=1.0),
                _record(2, "init", status="failed", reason="r"),
                _record(3, "bo", f=1.0),
                _record(4, "bo", status="failed", reason="r"),
                _record(5, "bo", status="abandoned", reason="budget"),
            ],
            stop_reason="budget",
            bindings=(_objective("f"),),
        )

        assert result.evaluations == {"x0": 1, "init": 2, "bo": 2, "failed": 2}

    def test_empty_run(self):
        result = BORunResult([], "budget", (_objective("f"),))

        assert result.evaluations == {"x0": 0, "init": 0, "bo": 0, "failed": 0}
        assert result.best is None
        assert result.pareto_front == []
        assert not result.feasible


class TestRunResultBestSingleObjective:
    BINDINGS = (_objective("f"), _constraint("c", bound=1.0))

    def _result(self, records, bindings=BINDINGS):
        return BORunResult(records, "budget", bindings)

    def test_best_feasible_record_minimises_the_objective(self):
        records = [
            _record(0, f=5.0, c=0.5),
            _record(1, f=3.0, c=0.9),
            _record(2, f=1.0, c=2.0),
        ]

        result = self._result(records)

        assert result.best is records[1]
        assert result.feasible
        assert result.pareto_front == []

    def test_maximisation_picks_the_largest_feasible_value(self):
        bindings = (_objective("f", minimize=False), _constraint("c", bound=1.0))
        records = [
            _record(0, f=5.0, c=0.5),
            _record(1, f=9.0, c=1.5),
            _record(2, f=7.0, c=0.9),
        ]

        assert self._result(records, bindings).best is records[2]

    def test_a_tie_goes_to_the_earliest_record(self):
        records = [_record(0, f=4.0, c=0.0), _record(1, f=3.0, c=0.0)]
        records.append(_record(2, f=3.0, c=0.0))

        assert self._result(records).best is records[1]

    def test_failed_and_abandoned_records_are_never_best(self):
        records = [
            _record(0, status="failed", reason="r"),
            _record(1, f=2.0, c=0.0),
            _record(2, status="abandoned", reason="budget"),
        ]

        assert self._result(records).best is records[1]

    def test_without_a_feasible_record_the_least_violation_wins(self):
        records = [
            _record(0, f=1.0, c=2.0),
            _record(1, f=9.0, c=1.5),
            _record(2, f=0.0, c=3.0),
        ]

        result = self._result(records)

        assert result.best is records[1]
        assert not result.feasible

    def test_violation_is_scaled_and_net_of_tolerance(self):
        bindings = (
            _objective("f"),
            _constraint("small", bound=0.0, tolerance=0.5, scale=1.0),
            _constraint("big", bound=0.0, scale=100.0),
        )
        records = [
            _record(0, f=0.0, small=2.5, big=0.0),
            _record(1, f=0.0, small=0.0, big=20.0),
            _record(2, f=0.0, small=0.5, big=60.0),
        ]

        # scaled violations: 2.0, 0.2 and 0.6
        assert self._result(records, bindings).best is records[1]

    def test_least_violation_tie_goes_to_the_earliest_record(self):
        records = [_record(0, f=5.0, c=2.0), _record(1, f=1.0, c=2.0)]

        assert self._result(records).best is records[0]

    def test_nothing_completed_has_no_best(self):
        records = [
            _record(0, status="failed", reason="r"),
            _record(1, status="abandoned", reason="budget"),
        ]

        result = self._result(records)

        assert result.best is None
        assert not result.feasible

    def test_select_best_is_what_the_result_reports(self):
        records = [_record(0, f=2.0, c=0.0), _record(1, f=1.0, c=0.0)]

        assert select_best(records, self.BINDINGS) is self._result(records).best


class TestRunResultMultiObjective:
    BINDINGS = (
        _objective("f1", gemseo_name="f1_f2", index=0),
        _objective("f2", minimize=False, gemseo_name="f1_f2", index=1),
        _constraint("c", bound=0.0),
    )

    def _result(self, records):
        return BORunResult(records, "budget", self.BINDINGS)

    def test_front_keeps_non_dominated_feasible_records_in_order(self):
        records = [
            _record(0, f1=1.0, f2=1.0, c=0.0),
            _record(1, f1=2.0, f2=3.0, c=0.0),
            _record(2, f1=2.0, f2=2.0, c=0.0),
            _record(3, f1=3.0, f2=3.0, c=0.0),
            _record(4, f1=0.5, f2=0.5, c=-1.0),
            _record(5, f1=0.1, f2=10.0, c=1.0),
            _record(6, status="failed", reason="r"),
        ]

        result = self._result(records)

        assert result.pareto_front == [records[0], records[1], records[4]]

    def test_equal_objectives_are_not_dominated(self):
        records = [
            _record(0, f1=1.0, f2=1.0, c=0.0),
            _record(1, f1=1.0, f2=1.0, c=0.0),
        ]

        assert self._result(records).pareto_front == records

    def test_best_is_the_front_point_closest_to_the_ideal_point(self):
        records = [
            _record(0, f1=1.0, f2=1.0, c=0.0),
            _record(1, f1=2.0, f2=3.0, c=0.0),
            _record(4, f1=0.5, f2=0.5, c=0.0),
        ]

        result = self._result(records)

        # Normalised distances to the ideal point: 0.867, 1.0 and 1.0.
        assert result.best is records[0]
        assert result.feasible

    def test_a_single_front_point_is_its_own_best(self):
        records = [_record(0, f1=1.0, f2=1.0, c=0.0), _record(1, f1=2.0, f2=0.0, c=0.0)]

        result = self._result(records)

        assert result.pareto_front == [records[0]]
        assert result.best is records[0]

    def test_an_objective_constant_over_the_front_does_not_divide_by_zero(self):
        records = [
            _record(0, f1=1.0, f2=2.0, c=0.0),
            _record(1, f1=2.0, f2=2.0, c=0.0),
            _record(2, f1=0.0, f2=2.0, c=0.0),
        ]

        result = self._result(records)

        assert result.pareto_front == [records[2]]
        assert result.best is records[2]

    def test_without_a_feasible_record_the_front_is_empty(self):
        records = [
            _record(0, f1=1.0, f2=1.0, c=3.0),
            _record(1, f1=0.0, f2=9.0, c=2.0),
        ]

        result = self._result(records)

        assert result.pareto_front == []
        assert result.best is records[1]
        assert not result.feasible


def test_candidate_and_outcome_defaults():
    candidate = Candidate({"x": 0.5})
    outcome = TrialOutcome("failed")

    assert candidate.key is None
    assert outcome.metrics == {}
    assert outcome.reason is None
