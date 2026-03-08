import re

with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

pareto_code = """        if is_moo:
            pareto_front = client.get_pareto_frontier()
            if pareto_front:
                logger.warning(
                    "Multi-objective optimization completed. The Pareto frontier contains %d points. "
                    "Arbitrarily extracting the first lexicographical boundary point to update the problem's design space. "
                    "The full Pareto front should be exported via problem.database.",
                    len(pareto_front)
                )
                best_parameters = pareto_front[0][0]
            else:
                raise ValueError("Pareto frontier is empty")"""

start_idx = content.find("        if is_moo:\n            pareto_front = client.get_pareto_frontier()\n            if pareto_front:\n                best_parameters = pareto_front[0][0]\n            else:\n                raise ValueError(\"Pareto frontier is empty\")")
if start_idx != -1:
    end_idx = start_idx + len("        if is_moo:\n            pareto_front = client.get_pareto_frontier()\n            if pareto_front:\n                best_parameters = pareto_front[0][0]\n            else:\n                raise ValueError(\"Pareto frontier is empty\")")
    content = content[:start_idx] + pareto_code + content[end_idx:]

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
