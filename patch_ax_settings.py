with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    code = f.read()

code = code.replace(
"""        for cstr in problem.constraints:
            if cstr.f_type == 'ineq':
            cstr_name = cstr.name
            outcome_constraints.append(f"{cstr_name} <= 0.0")""",
"""        for cstr in problem.constraints:
            if cstr.f_type == 'ineq':
                cstr_name = cstr.name
                outcome_constraints.append(f"{cstr_name} <= 0.0")"""
)

code = code.replace(
"""                    for cstr in problem.constraints:
                        if cstr.f_type == 'ineq':
                            val = cstr.value
                        val = cstr.evaluate(x)
                        results[cstr.name] = float(np.max(val)) if isinstance(val, np.ndarray) else float(val)""",
"""                    for cstr in problem.constraints:
                        if cstr.f_type == 'ineq':
                            val = cstr.evaluate(x)
                            results[cstr.name] = float(np.max(val)) if isinstance(val, np.ndarray) else float(val)"""
)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(code)
