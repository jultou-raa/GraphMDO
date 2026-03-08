import re

with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

# Fix the classes that weren't fully removed
content = content.replace("class AxParameterBuilder:\n\"\"\"Builder class for Ax parameter configurations.\"\"\"\n\n@staticmethod\n", "")
content = content.replace("class AxObjectiveBuilder:\n\"\"\"Builder for Ax objective configurations.\"\"\"\n\n@staticmethod\n", "")
content = content.replace("class AxConstraintBuilder:\n\"\"\"Builder for Ax constraint configurations.\"\"\"\n\n@staticmethod\n", "")

content = content.replace("    @staticmethod\ndef build_from_ax_parameters", "def build_from_ax_parameters")

# I must just remove the class definitions directly!
lines = content.split('\n')
out = []
for line in lines:
    if line.strip() in [
        "class AxParameterBuilder:",
        '"""Builder class for Ax parameter configurations."""',
        "@staticmethod",
        "class AxObjectiveBuilder:",
        '"""Builder class for Ax objectives."""',
        '"""Builder for Ax objective configurations."""',
        "class AxConstraintBuilder:",
        '"""Builder class for Ax outcome constraints."""',
        '"""Builder for Ax constraint configurations."""'
    ]:
        continue
    out.append(line)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write('\n'.join(out))
