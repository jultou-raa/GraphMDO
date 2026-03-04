import openmdao.api as om
from typing import Any


class LocalEvaluator:
    """
    Evaluates the design parameters locally using an OpenMDAO Problem instance.

    Args:
        problem: An instantiated OpenMDAO Problem object.
    """

    def __init__(self, problem: om.Problem):
        self.problem = problem

    def evaluate(
        self, parameters: dict[str, Any], objectives: list[str]
    ) -> dict[str, float]:
        for name, val in parameters.items():
            self.problem.set_val(name, val)
        self.problem.run_model()

        results = {}
        for obj in objectives:
            results[obj] = (
                float(self.problem.get_val(obj)[0])
                if hasattr(self.problem.get_val(obj), "__iter__")
                else float(self.problem.get_val(obj))
            )
        return results
