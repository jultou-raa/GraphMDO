with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

content = content.replace('logger.error("Failed to evaluate point: %s\n%s", e, traceback.format_exc())', 'logger.error("Failed to evaluate point: %s\\n%s", e, traceback.format_exc())')

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
