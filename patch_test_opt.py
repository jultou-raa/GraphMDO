with open("tests/test_optimizer.py", "r") as f:
    content = f.read()

content = content.replace("    LocalEvaluator,\n", "")

with open("tests/test_optimizer.py", "w") as f:
    f.write(content)
