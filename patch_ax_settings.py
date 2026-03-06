# Fix `ChoiceParameterConfig` is not defined!
# I removed it in ruff check fix earlier in `patch_tests.py` using `ruff check --fix`!!
# Ruff removed it from imports because it WAS unused before I added the fallback loop back!
# Let's add `ChoiceParameterConfig` to imports in `ax_algo_lib.py`!
with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    text = f.read()

text = text.replace(
"""from ax.api.configs import RangeParameterConfig""",
"""from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig"""
)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(text)
