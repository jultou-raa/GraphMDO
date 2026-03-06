with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    text = f.read()

# Fix ValueError: Algorithm Ax_Bayesian is not adapted to the problem, it does not handle integer variables.
# In `ax_algo_lib.py`, `ALGORITHM_INFOS["Ax_Bayesian"]` has `handle_integer_variables=False`!
# Ah! I changed it to True earlier but then I might have overwritten it or it was False in my previous edits!
# Let me just set it to True!
text = text.replace("handle_integer_variables=False", "handle_integer_variables=True")

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(text)
