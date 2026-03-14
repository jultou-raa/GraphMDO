with open("src/services/execution/main.py", "r") as f:
    content = f.read()

target = """    results = {}
    for obj in objectives:
        val = out_data.get(obj)
        if val is None:
            results[obj] = 0.0
        else:
            results[obj] = val
    return results"""

replacement = """    return {
        obj: 0.0 if (val := out_data.get(obj)) is None else val
        for obj in objectives
    }"""

new_content = content.replace(target, replacement)
with open("src/services/execution/main.py", "w") as f:
    f.write(new_content)
