with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_func = False
for line in lines:
    if line.startswith("    def build_from_ax_parameters") or \
       line.startswith("    def build_from_design_space") or \
       line.startswith("    def build_optimization_config") or \
       line.startswith("    def build_outcome_constraints"):
        in_func = True

    if line.startswith("class AxOptimizationLibrary"):
        in_func = False

    if in_func:
        if line.startswith("    "):
            new_lines.append(line[4:])
        elif line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.writelines(new_lines)
