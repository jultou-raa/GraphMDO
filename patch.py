with open("README.md", "r") as f:
    content = f.read()

content = content.replace("KADMOS for semantic formulation and exporting to CMDOWS", "GEMSEO for semantic formulation and execution")
content = content.replace("execution is handled by OpenMDAO", "execution is handled natively by GEMSEO")
content = content.replace("translates the graph topology into an executable [OpenMDAO](https://openmdao.org/) problem.", "translates the graph topology into an executable [GEMSEO](https://gemseo.readthedocs.io/) MDO formulation.")
content = content.replace("KADMOS multi-objective targets", "GEMSEO multi-objective targets")
content = content.replace("OpenMDAO System", "GEMSEO Problem")
content = content.replace("OpenMDAO problem", "GEMSEO problem")
content = content.replace("Build OpenMDAO Problem from Graph", "Build GEMSEO Problem from Graph")
content = content.replace("from KADMOS graph", "from the graph schema")

with open("README.md", "w") as f:
    f.write(content)
