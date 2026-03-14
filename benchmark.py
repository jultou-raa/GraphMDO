import timeit

setup = """
out_data = {f"obj_{i}": i for i in range(100)}
objectives = [f"obj_{i}" for i in range(100)] + [f"missing_{i}" for i in range(10)]
"""

code_original = """
results = {}
for obj in objectives:
    val = out_data.get(obj)
    if val is None:
        results[obj] = 0.0
    else:
        results[obj] = val
"""

code_optimized = """
results = {obj: val if (val := out_data.get(obj)) is not None else 0.0 for obj in objectives}
"""

print("Original:", timeit.timeit(code_original, setup=setup, number=100000))
print("Optimized:", timeit.timeit(code_optimized, setup=setup, number=100000))
