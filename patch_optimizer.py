import re

with open("src/mdo_framework/optimization/optimizer.py", "r") as f:
    content = f.read()

# Remove import openmdao.api as om
content = re.sub(r"import openmdao\.api as om\n*", "", content)

# Remove LocalEvaluator class definition
content = re.sub(
    r"class LocalEvaluator:\n(?: {4}.*\n|\n)+?(?=class RemoteEvaluator:)", "", content
)

with open("src/mdo_framework/optimization/optimizer.py", "w") as f:
    f.write(content)
