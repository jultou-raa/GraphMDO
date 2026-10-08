"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Real GEMSEO MDAs over small pure-Python tools.
"""

import functools
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from pydantic import ValidationError

from mdo_framework.core.errors import (
    EvaluationError,
    MDANotConvergedError,
    ToolExecutionError,
)
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.mda import MDASettings, StrictMDAChain, build_mda
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.schema import (
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolSpec,
)

X1 = {"x": np.array([1.0])}


def two_tool_schema(
    *, thread_safe: bool = False, deterministic: bool = True
) -> StudySchema:
    """x, y2 -> T1 -> y1 and y1 -> T2 -> y2: a two-tool cycle."""
    options = {"thread_safe": thread_safe, "deterministic": deterministic}
    return StudySchema(
        variables=[
            RangeVar(name="x", lower=0.0, upper=10.0),
            StateVar(name="y1"),
            StateVar(name="y2"),
        ],
        tools=[
            ToolSpec(name="T1", inputs=["x", "y2"], outputs=["y1"], **options),
            ToolSpec(name="T2", inputs=["y1"], outputs=["y2"], **options),
        ],
    )


# Fixed point y1 = (x + 0.5) / 0.75, so y1 = y2 = 2 at x = 1.
CONTRACTION = {
    "T1": lambda x, y2: x + 0.5 * y2,
    "T2": lambda y1: 0.5 * y1 + 1.0,
}
NO_FIXED_POINT = {
    "T1": lambda x, y2: y2 + 1.0 + x,
    "T2": lambda y1: y1 + 1.0,
}


def sellar_schema(*, thread_safe: bool = True) -> StudySchema:
    """Scalar Sellar: Sellar1 and Sellar2 are coupled, System is not."""
    return StudySchema(
        variables=[
            RangeVar(name="x", lower=0.0, upper=10.0),
            RangeVar(name="z1", lower=-10.0, upper=10.0),
            RangeVar(name="z2", lower=0.0, upper=10.0),
            StateVar(name="y1"),
            StateVar(name="y2"),
            StateVar(name="obj"),
            StateVar(name="c1"),
            StateVar(name="c2"),
        ],
        tools=[
            ToolSpec(
                name="Sellar1",
                inputs=["x", "z1", "z2", "y2"],
                outputs=["y1"],
                thread_safe=thread_safe,
            ),
            ToolSpec(
                name="Sellar2",
                inputs=["z1", "z2", "y1"],
                outputs=["y2"],
                thread_safe=thread_safe,
            ),
            ToolSpec(
                name="System",
                inputs=["x", "z1", "z2", "y1", "y2"],
                outputs=["obj", "c1", "c2"],
            ),
        ],
    )


def sellar1(x, z1, z2, y2):
    return z1**2 + z2 + x - 0.2 * y2


def sellar2(z1, z2, y1):
    return abs(y1) ** 0.5 + z1 + z2


def sellar_system(x, z1, z2, y1, y2):
    return {
        "obj": x**2 + z2 + y1 + np.exp(-y2),
        "c1": 3.16 - y1,
        "c2": y2 - 24.0,
    }


SELLAR = {"Sellar1": sellar1, "Sellar2": sellar2, "System": sellar_system}
SELLAR_POINT = {
    "x": np.array([1.0]),
    "z1": np.array([5.0]),
    "z2": np.array([2.0]),
}


class ConcurrencyProbe:
    """Counts how many wrapped tools run at the same time."""

    def __init__(self, pause: float = 0.01) -> None:
        self.pause = pause
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def probed(**inputs: Any) -> Any:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(self.pause)
                return function(**inputs)
            finally:
                with self._lock:
                    self.active -= 1

        return probed

    def wrap_all(self, registry: dict[str, Callable[..., Any]]) -> dict[str, Any]:
        return {name: self.wrap(function) for name, function in registry.items()}


def build(
    schema: StudySchema,
    registry: dict[str, Callable[..., Any]],
    settings: MDASettings | None = None,
) -> StrictMDAChain:
    return GraphProblemBuilder(schema).build_problem(registry, settings)


def scalar(output: Any, name: str) -> float:
    return float(np.asarray(output[name]).flat[0])


class TestMDASettings:
    def test_defaults_run_the_couplings_sequentially_with_gemseo_tolerances(self):
        settings = MDASettings()

        assert settings.inner_mda_name == "MDAGaussSeidel"
        assert settings.tolerance == 1e-6
        assert settings.max_mda_iter == 20
        assert settings.max_consecutive_unsuccessful_iterations == 8
        assert settings.n_processes == 1
        assert settings.accept_tolerance is None

    def test_settings_are_immutable_and_reject_unknown_fields(self):
        with pytest.raises(ValidationError):
            MDASettings().tolerance = 1e-3
        with pytest.raises(ValidationError):
            MDASettings(use_threading=True)

    @pytest.mark.parametrize(
        "field", ["tolerance", "max_mda_iter", "n_processes", "accept_tolerance"]
    )
    def test_numbers_must_be_positive(self, field):
        with pytest.raises(ValidationError):
            MDASettings(**{field: 0})

    def test_unknown_inner_mda_is_rejected(self):
        with pytest.raises(ValidationError):
            MDASettings(inner_mda_name="MDAChain")

    @pytest.mark.parametrize("name", ["MDAGaussSeidel", "MDANewtonRaphson"])
    def test_parallel_execution_needs_the_jacobi_algorithm(self, name):
        with pytest.raises(ValidationError, match="MDAJacobi"):
            MDASettings(inner_mda_name=name, n_processes=2)

    def test_jacobi_accepts_several_processes(self):
        settings = MDASettings(inner_mda_name="MDAJacobi", n_processes=4)

        assert settings.n_processes == 4

    def test_accept_tolerance_cannot_be_tighter_than_tolerance(self):
        with pytest.raises(ValidationError, match="accept_tolerance"):
            MDASettings(tolerance=1e-4, accept_tolerance=1e-5)

    def test_accept_tolerance_may_equal_tolerance(self):
        assert MDASettings(tolerance=1e-4, accept_tolerance=1e-4)


class TestSettingsPlumbing:
    def test_default_inner_mda_is_gauss_seidel_with_the_default_values(self):
        chain = build(two_tool_schema(), CONTRACTION)

        (inner,) = chain.inner_mdas
        assert isinstance(chain, StrictMDAChain)
        assert isinstance(inner, MDAGaussSeidel)
        assert inner.settings.tolerance == 1e-6
        assert inner.settings.max_mda_iter == 20
        assert inner.settings.max_consecutive_unsuccessful_iterations == 8

    def test_configured_values_reach_the_inner_mda(self):
        settings = MDASettings(
            tolerance=1e-8,
            max_mda_iter=7,
            max_consecutive_unsuccessful_iterations=3,
        )

        (inner,) = build(two_tool_schema(), CONTRACTION, settings).inner_mdas

        assert inner.settings.tolerance == 1e-8
        assert inner.settings.max_mda_iter == 7
        assert inner.settings.max_consecutive_unsuccessful_iterations == 3

    def test_jacobi_gets_its_process_count_explicitly(self):
        sequential = MDASettings(inner_mda_name="MDAJacobi")
        parallel = MDASettings(inner_mda_name="MDAJacobi", n_processes=2)
        schema = two_tool_schema(thread_safe=True)

        (inner_sequential,) = build(schema, CONTRACTION, sequential).inner_mdas
        (inner_parallel,) = build(schema, CONTRACTION, parallel).inner_mdas

        assert isinstance(inner_sequential, MDAJacobi)
        assert inner_sequential.settings.n_processes == 1
        assert inner_parallel.settings.n_processes == 2

    def test_newton_raphson_never_falls_back_to_one_thread_per_cpu(self):
        settings = MDASettings(inner_mda_name="MDANewtonRaphson")

        (inner,) = build(two_tool_schema(), CONTRACTION, settings).inner_mdas

        assert isinstance(inner, MDANewtonRaphson)
        assert inner.settings.n_processes == 1

    def test_build_mda_takes_disciplines_and_default_settings(self):
        problem = build(two_tool_schema(), CONTRACTION)

        chain = build_mda(problem.disciplines)

        assert isinstance(chain.inner_mdas[0], MDAGaussSeidel)


class TestConvergenceIsEnforced:
    def test_without_a_fixed_point_the_evaluation_raises(self):
        problem = build(two_tool_schema(), NO_FIXED_POINT)

        with pytest.raises(MDANotConvergedError) as exc_info:
            problem.execute(X1)

        error = exc_info.value
        assert error.code == "MDA_NOT_CONVERGED"
        assert isinstance(error, EvaluationError)
        assert isinstance(error, ValueError)
        assert "MDA did not converge: residual " in str(error)
        assert "> 1.0e-06" in str(error)
        assert "(couplings: y1, y2)" in str(error)

    def test_a_non_converged_point_raises_every_time(self):
        problem = build(two_tool_schema(), NO_FIXED_POINT)

        for _ in range(3):
            with pytest.raises(MDANotConvergedError):
                problem.execute(X1)

    def test_local_evaluator_raises_instead_of_returning_the_last_iterate(self):
        builder = GraphProblemBuilder(two_tool_schema())
        evaluator = LocalEvaluator(
            builder.build_problem(NO_FIXED_POINT), builder.variable_specs
        )

        with pytest.raises(MDANotConvergedError, match="y1, y2"):
            evaluator.evaluate({"x": 1.0}, ["y1", "y2"])

    def test_a_contraction_converges_under_the_defaults(self):
        output = build(two_tool_schema(), CONTRACTION).execute(X1)

        assert scalar(output, "y1") == pytest.approx(2.0, abs=1e-4)
        assert scalar(output, "y2") == pytest.approx(2.0, abs=1e-4)

    def test_too_few_iterations_fail_unless_accept_tolerance_allows_them(self):
        few = MDASettings(max_mda_iter=3)
        loose = MDASettings(max_mda_iter=3, accept_tolerance=0.5)

        with pytest.raises(MDANotConvergedError):
            build(two_tool_schema(), CONTRACTION, few).execute(X1)
        output = build(two_tool_schema(), CONTRACTION, loose).execute(X1)

        assert scalar(output, "y1") == pytest.approx(2.0, abs=0.5)

    def test_the_message_names_the_limit_in_force(self):
        settings = MDASettings(max_mda_iter=1, accept_tolerance=1e-3)

        with pytest.raises(MDANotConvergedError, match="> 1.0e-03"):
            build(two_tool_schema(), CONTRACTION, settings).execute(X1)

    def test_an_acyclic_graph_has_no_residual_to_check(self):
        schema = StudySchema(
            variables=[
                RangeVar(name="x", lower=0.0, upper=10.0),
                StateVar(name="a"),
                StateVar(name="b"),
            ],
            tools=[
                ToolSpec(name="A", inputs=["x"], outputs=["a"]),
                ToolSpec(name="B", inputs=["a"], outputs=["b"]),
            ],
        )
        problem = build(schema, {"A": lambda x: x + 1.0, "B": lambda a: 2 * a})

        assert problem.inner_mdas == []
        assert scalar(problem.execute(X1), "b") == 4.0

    def test_the_convergence_check_runs_in_every_optimizer_entry_point(self):
        problem = build(two_tool_schema(), NO_FIXED_POINT)
        space = DesignSpace()
        space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=X1["x"])
        scenario = create_scenario(
            [problem],
            formulation_name="MDF",
            objective_name="y1",
            design_space=space,
        )

        with pytest.raises(MDANotConvergedError):
            scenario.execute(algo_name="SLSQP", max_iter=2)


class TestCouplingsRunSequentially:
    def test_default_settings_never_run_two_tools_at_once(self):
        probe = ConcurrencyProbe()
        problem = build(sellar_schema(thread_safe=False), probe.wrap_all(SELLAR))

        problem.execute(SELLAR_POINT)

        assert probe.max_active == 1

    def test_the_probe_sees_the_overlap_of_a_parallel_jacobi(self):
        probe = ConcurrencyProbe()
        settings = MDASettings(inner_mda_name="MDAJacobi", n_processes=2)
        problem = build(sellar_schema(), probe.wrap_all(SELLAR), settings)

        problem.execute(SELLAR_POINT)

        assert probe.max_active == 2


class TestParallelJacobiFailures:
    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("solver crashed"),
            KeyError("missing"),
            OSError("disk full"),
            TypeError("bad operand"),
            subprocess.CalledProcessError(1, "solver"),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_a_failing_coupled_tool_makes_the_evaluation_raise(self, error):
        def failing(y1):
            raise error

        registry = {**CONTRACTION, "T2": failing}
        settings = MDASettings(inner_mda_name="MDAJacobi", n_processes=2)
        problem = build(two_tool_schema(thread_safe=True), registry, settings)

        with pytest.raises(ToolExecutionError) as exc_info:
            problem.execute(X1)

        assert exc_info.value.tool == "T2"
        assert type(error).__name__ in str(exc_info.value)


class TestThreadSafetyGate:
    @pytest.fixture
    def parallel(self) -> MDASettings:
        return MDASettings(inner_mda_name="MDAJacobi", n_processes=2)

    def test_coupled_tools_must_declare_thread_safety_to_run_in_parallel(
        self, parallel
    ):
        builder = GraphProblemBuilder(two_tool_schema(thread_safe=False))

        with pytest.raises(StudyValidationError) as exc_info:
            builder.build_problem(CONTRACTION, parallel)

        errors = exc_info.value.report.errors
        assert [(error.code, error.names) for error in errors] == [
            ("TOOL_NOT_THREAD_SAFE", ("T1",)),
            ("TOOL_NOT_THREAD_SAFE", ("T2",)),
        ]
        assert "thread_safe" in str(exc_info.value)

    def test_thread_safe_coupled_tools_are_accepted(self):
        # Jacobi contracts at half the rate of Gauss-Seidel on this graph.
        settings = MDASettings(
            inner_mda_name="MDAJacobi", n_processes=2, max_mda_iter=60
        )
        problem = build(two_tool_schema(thread_safe=True), CONTRACTION, settings)

        assert scalar(problem.execute(X1), "y1") == pytest.approx(2.0, abs=1e-4)

    def test_tools_outside_the_coupled_group_need_no_declaration(self, parallel):
        schema = sellar_schema(thread_safe=True)

        problem = build(schema, SELLAR, parallel)

        assert schema.tool("System").thread_safe is False
        assert isinstance(problem, StrictMDAChain)

    def test_only_the_undeclared_coupled_tool_is_reported(self, parallel):
        schema = sellar_schema(thread_safe=True)
        tools = [
            tool.model_copy(update={"thread_safe": tool.name != "Sellar2"})
            for tool in schema.tools
        ]
        schema = schema.model_copy(update={"tools": tools})

        with pytest.raises(StudyValidationError) as exc_info:
            build(schema, SELLAR, parallel)

        assert [error.names for error in exc_info.value.report.errors] == [("Sellar2",)]

    def test_the_gate_does_not_apply_to_sequential_execution(self):
        assert build(two_tool_schema(thread_safe=False), CONTRACTION)

    def test_registry_errors_are_reported_together_with_the_gate(self, parallel):
        builder = GraphProblemBuilder(two_tool_schema(thread_safe=False))

        with pytest.raises(StudyValidationError) as exc_info:
            builder.build_problem({"T1": CONTRACTION["T1"]}, parallel)

        codes = [error.code for error in exc_info.value.report.errors]
        assert codes == [
            "UNREGISTERED_TOOL",
            "TOOL_NOT_THREAD_SAFE",
            "TOOL_NOT_THREAD_SAFE",
        ]


class TestNonDeterministicTools:
    @staticmethod
    def counted(registry: dict[str, Callable[..., Any]]):
        calls: list[str] = []

        def count(name: str, function: Callable[..., Any]):
            @functools.wraps(function)
            def counted_function(**inputs: Any) -> Any:
                calls.append(name)
                return function(**inputs)

            return counted_function

        counted_registry = {name: count(name, fn) for name, fn in registry.items()}
        return counted_registry, calls

    @staticmethod
    def calls_per_execution(problem, calls: list[str], n: int = 5) -> list[int]:
        per_execution = []
        for _ in range(n):
            calls.clear()
            problem.execute(X1)
            per_execution.append(len(calls))
        return per_execution

    def test_deterministic_graph_replays_the_cache(self):
        registry, calls = self.counted(CONTRACTION)
        problem = build(two_tool_schema(), registry)

        per_execution = self.calls_per_execution(problem, calls)

        assert per_execution[0] > 0
        assert per_execution[1:] == [0, 0, 0, 0]

    def test_acyclic_graph_calls_a_stochastic_tool_every_time(self):
        schema = StudySchema(
            variables=[RangeVar(name="x", lower=0.0, upper=10.0), StateVar(name="a")],
            tools=[
                ToolSpec(name="A", inputs=["x"], outputs=["a"], deterministic=False)
            ],
        )
        registry, calls = self.counted({"A": lambda x: x})
        problem = build(schema, registry)

        assert self.calls_per_execution(problem, calls) == [1] * 5

    def test_coupled_graph_calls_a_stochastic_tool_every_time(self):
        registry, calls = self.counted(CONTRACTION)
        problem = build(two_tool_schema(deterministic=False), registry)

        per_execution = self.calls_per_execution(problem, calls)

        assert min(per_execution) > 0
        assert len(set(per_execution)) == 1

    def test_deterministic_neighbours_keep_their_cache(self):
        schema = StudySchema(
            variables=[
                RangeVar(name="x", lower=0.0, upper=10.0),
                StateVar(name="a"),
                StateVar(name="b"),
            ],
            tools=[
                ToolSpec(name="A", inputs=["x"], outputs=["a"]),
                ToolSpec(name="B", inputs=["a"], outputs=["b"], deterministic=False),
            ],
        )
        registry, calls = self.counted({"A": lambda x: x, "B": lambda a: a})
        problem = build(schema, registry)

        self.calls_per_execution(problem, calls)

        assert calls == ["B"]


class TestNewtonRaphsonAndOptimization:
    def test_newton_raphson_solves_the_scalar_sellar_couplings(self):
        settings = MDASettings(inner_mda_name="MDANewtonRaphson")
        problem = build(sellar_schema(), SELLAR, settings)

        output = problem.execute(SELLAR_POINT)

        assert scalar(output, "y1") == pytest.approx(25.5883, rel=1e-4)
        assert scalar(output, "y2") == pytest.approx(12.0585, rel=1e-4)

    def test_slsqp_runs_on_the_translated_graph(self):
        problem = build(sellar_schema(), SELLAR)
        space = DesignSpace()
        space.add_variable(
            "x", lower_bound=0.0, upper_bound=10.0, value=np.array([1.0])
        )
        space.add_variable(
            "z1", lower_bound=-10.0, upper_bound=10.0, value=np.array([5.0])
        )
        space.add_variable(
            "z2", lower_bound=0.0, upper_bound=10.0, value=np.array([2.0])
        )
        scenario = create_scenario(
            [problem],
            formulation_name="MDF",
            objective_name="obj",
            design_space=space,
        )

        scenario.set_differentiation_method("finite_differences")

        scenario.execute(algo_name="SLSQP", max_iter=3)

        assert scenario.optimization_result.f_opt is not None
