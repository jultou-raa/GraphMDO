with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

import re

# Fix Multi-Objective Directional Logic & Hypervolume Reference Points
# In build_optimization_config:
#                minimize = obj.get("minimize", True)
# ->             minimize = obj.get("minimize", problem.minimize_objective)
content = content.replace('minimize = obj.get("minimize", True)', 'minimize = obj.get("minimize", problem.minimize_objective)')

# Remove threshold logic:
#                if threshold is None:
#                    # Temporary fallback, can be adjusted in step 2.
#                    threshold = 1e6 if minimize else -1e6
#
#                ax_objs.append(
#                    Objective(
#                        metric=MapMetric(name=metric_name),
#                        minimize=minimize,
#                    )
#                )
#
# Wait, ax_objs is built by mapping. If threshold is provided by the user, we should add objective thresholds. BUT if threshold is None, we DON'T add arbitrary boundaries!
# Let's see how objective thresholds are passed to Ax.
# They are passed to `MultiObjectiveOptimizationConfig`!
# `objective_thresholds=[ObjectiveThreshold(metric=MapMetric(name=metric_name), bound=threshold, relative=False)]`

# Wait, the current code just appends `Objective` to `ax_objs` and DOES NOT set `objective_thresholds` in `MultiObjectiveOptimizationConfig`!
# Let's fix this to set thresholds if provided!

moo_config_code = """
        if ax_objectives and len(ax_objectives) > 1:
            ax_objs = []
            objective_thresholds = []
            for obj in ax_objectives:
                metric_name = obj["name"]
                minimize = obj.get("minimize", problem.minimize_objective)
                threshold = obj.get("threshold", None)

                ax_objs.append(
                    Objective(
                        metric=MapMetric(name=metric_name),
                        minimize=minimize,
                    )
                )

                if threshold is not None:
                    from ax.core.objective import ObjectiveThreshold
                    objective_thresholds.append(
                        ObjectiveThreshold(
                            metric=MapMetric(name=metric_name),
                            bound=float(threshold),
                            relative=False,
                            op=ComparisonOp.LEQ if minimize else ComparisonOp.GEQ,
                        )
                    )

            return MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=ax_objs),
                objective_thresholds=objective_thresholds if objective_thresholds else None,
                outcome_constraints=ax_outcome_constraints,
            )
"""

# Replace the MOO block:
start_idx = content.find("if ax_objectives and len(ax_objectives) > 1:")
end_idx = content.find("else:", start_idx)

content = content[:start_idx] + moo_config_code.strip() + "\n        " + content[end_idx:]

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
