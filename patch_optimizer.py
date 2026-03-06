with open("src/mdo_framework/optimization/optimizer.py", "r") as f:
    code = f.read()

import re

# Just fix the duplicate imports at line 33 and remove unused ones,
# and add json back!

code = re.sub(r'import warnings\nfrom typing import Any, Protocol\n\nimport httpx\nfrom ax\..*?\nfrom ax\..*?\nfrom ax\..*?\nfrom ax\..*?\n    GenerationStep,\n    GenerationStrategy,\n\)\nfrom botorch\.acquisition\.logei import qLogNoisyExpectedImprovement\n', '', code, flags=re.MULTILINE)
code = re.sub(r'import warnings\nfrom typing import Any, Protocol\n\nimport httpx\n', '', code)

with open("src/mdo_framework/optimization/optimizer.py", "w") as f:
    f.write(code)
