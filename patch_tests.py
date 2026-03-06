# Fix the objective evaluation!
# In `ax_algo_lib.py`:
# `results[obj_name] = float(problem.objective.value[0])`
# But if it's evaluated natively, `evaluate_functions` sets `problem.objective.value`.
# BUT wait! The exception string `Failed to evaluate point:` is completely empty because I logged `logger.error(f"Failed to evaluate point: {e}")` but `{e}` was empty?
# Or `traceback` wasn't printed?
# If I use `float(problem.objective.value[0])`, it throws `TypeError: 'NoneType' object is not subscriptable` if it's None.
# Let me change it back to just evaluate it directly.
with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    text = f.read()

text = text.replace(
"""                    results[obj_name] = float(problem.objective.value[0])""",
"""                    obj_val = problem.objective.value
                    results[obj_name] = float(obj_val[0]) if isinstance(obj_val, np.ndarray) else float(obj_val)"""
)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(text)
