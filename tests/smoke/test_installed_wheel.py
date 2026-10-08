"""Smoke-test the installed distribution against its supported Ax/BoTorch pair."""

from importlib.metadata import version

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.schema import RangeVar, StateVar, StudySchema, ToolSpec


def test_installed_wheel_runs_a_real_optimization(tmp_path, monkeypatch) -> None:
    assert version("ax-platform") == "1.2.4"
    assert version("botorch") == "0.17.2"
    monkeypatch.chdir(tmp_path)

    def paraboloid(x: float, y: float) -> dict[str, float]:
        return {
            "f_xy": (x - 3) ** 2 + x * y + (y + 4) ** 2 - 3,
            "c_xy": x - y,
        }

    schema = StudySchema(
        variables=[
            *(RangeVar(name=name, lower=-10.0, upper=10.0) for name in ("x", "y")),
            *(StateVar(name=name) for name in ("f_xy", "c_xy")),
        ],
        tools=[ToolSpec(name="P", inputs=["x", "y"], outputs=["f_xy", "c_xy"])],
    )
    builder = GraphProblemBuilder(schema)
    problem = builder.build_problem({"P": paraboloid})
    optimizer = BayesianOptimizer(
        LocalEvaluator(problem, builder.variable_specs),
        [
            {"name": name, "type": "range", "bounds": [-10.0, 10.0]}
            for name in ("x", "y")
        ],
        [{"name": "f_xy"}],
        [{"name": "c_xy", "op": "<=", "bound": 0.0}],
    )

    result = optimizer.optimize(n_steps=6, n_init=3)

    assert len(result["history"]) >= 2
