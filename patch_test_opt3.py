with open("tests/test_optimizer.py", "r") as f:
    content = f.read()

content = content.replace(
    "from mdo_framework.optimization.optimizer import (\nfrom mdo_framework.core.evaluators import LocalEvaluator",
    "from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator\nfrom mdo_framework.core.evaluators import LocalEvaluator",
)

with open("tests/test_optimizer.py", "w") as f:
    f.write(content)
