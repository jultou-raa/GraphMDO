with open("README.md", "r") as f:
    content = f.read()

content = content.replace("specifically utilizing GEMSEO for semantic formulation and execution. The execution is handled natively by GEMSEO", "specifically utilizing GEMSEO for semantic formulation and execution. Execution is handled natively by GEMSEO")
content = content.replace("an GEMSEO", "a GEMSEO")

with open("README.md", "w") as f:
    f.write(content)
