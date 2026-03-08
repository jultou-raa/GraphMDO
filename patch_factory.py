import re

with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

# Replace the module-level functions with a class!

class_def = """class AxConfigurationFactory:
    \"\"\"Factory for building Ax platform configurations from GEMSEO inputs.\"\"\"

    @staticmethod
    def build_from_ax_parameters(
        ax_parameters: list[AxParameterDict],
    ) -> list[RangeParameterConfig | ChoiceParameterConfig]:
        ax_params = []
        for p in ax_parameters:
            if p["type"] == "range":
                if len(p["bounds"]) != 2:
                    raise ValueError(f"Range parameter {p['name']} requires 2 bounds.")
                ax_params.append(
                    RangeParameterConfig(
                        name=p["name"],
                        bounds=(p["bounds"][0], p["bounds"][1]),
                        parameter_type=p.get("value_type", "float"),
                    )
                )
            elif p["type"] == "choice":
                if not p.get("values"):
                    raise ValueError(
                        f"Choice parameter {p['name']} requires a list of values."
                    )
                ax_params.append(
                    ChoiceParameterConfig(
                        name=p["name"],
                        values=p["values"],
                        parameter_type=p.get(
                            "value_type",
                            "str" if isinstance(p["values"][0], str) else "float",
                        ),
                    )
                )
        return ax_params

    @staticmethod
    def build_from_design_space(
        design_space: Any, normalize: bool
    ) -> list[RangeParameterConfig]:
        ax_params = []
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            l_b, u_b, _, _ = get_value_and_bounds(
                design_space, var_name, normalize=normalize
            )
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                ax_params.append(
                    RangeParameterConfig(
                        name=param_name, bounds=(float(l_b[i]), float(u_b[i]))
                    )
                )
        return ax_params

    @staticmethod
    def build_optimization_config(
        ax_objectives: list[AxObjectiveDict] | None,
        problem: OptimizationProblem,
        ax_outcome_constraints: list[OutcomeConstraint],
    ) -> OptimizationConfig | MultiObjectiveOptimizationConfig:
        if ax_objectives and len(ax_objectives) > 1:
            ax_objs = []
            for obj in ax_objectives:
                metric_name = obj["name"]
                minimize = obj.get("minimize", True)
                threshold = obj.get("threshold", None)

                if threshold is None:
                    # Temporary fallback, can be adjusted in step 2.
                    threshold = 1e6 if minimize else -1e6

                ax_objs.append(
                    Objective(
                        metric=MapMetric(name=metric_name),
                        minimize=minimize,
                    )
                )

            return MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=ax_objs),
                outcome_constraints=ax_outcome_constraints,
            )
        else:
            if ax_objectives and len(ax_objectives) == 1:
                metric_name = ax_objectives[0]["name"]
                minimize = ax_objectives[0].get("minimize", problem.minimize_objective)
            else:
                metric_name = (
                    problem.objective.name[0]
                    if isinstance(problem.objective.name, list)
                    else problem.objective.name
                )
                minimize = problem.minimize_objective

            return OptimizationConfig(
                objective=Objective(
                    metric=MapMetric(name=metric_name), minimize=minimize
                ),
                outcome_constraints=ax_outcome_constraints,
            )

    @staticmethod
    def build_outcome_constraints(
        constraints: list[Any],
    ) -> list[OutcomeConstraint]:
        ax_outcome_constraints = []
        for c in constraints:
            if c.f_type == "ineq":
                ax_outcome_constraints.append(
                    OutcomeConstraint(
                        metric=MapMetric(name=c.name),
                        op=ComparisonOp.LEQ,
                        bound=0.0,
                        relative=False,
                    )
                )
        return ax_outcome_constraints
"""


start_idx = content.find("def build_from_ax_parameters(")
end_idx = content.find("class AxOptimizationLibrary")
content = content[:start_idx] + class_def + "\n\n" + content[end_idx:]

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
