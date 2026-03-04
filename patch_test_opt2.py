with open("tests/test_optimizer.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if (
        line.strip() in ["BayesianOptimizer,", "RemoteEvaluator,", ")"]
        and "from" not in line
    ):
        continue
    new_lines.append(line)

with open("tests/test_optimizer.py", "w") as f:
    f.writelines(new_lines)
