import timeit

setup = """
out_data = {f"obj_{i}": (i if i % 10 != 0 else None) for i in range(100)}
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

code_dict_comp_ternary = """
results = {obj: 0.0 if (v := out_data.get(obj)) is None else v for obj in objectives}
"""

code_dict_comp_or = """
results = {obj: out_data.get(obj) or 0.0 for obj in objectives}
"""

print("Original:", timeit.timeit(code_original, setup=setup, number=100000))
print("Dict Comp (walrus):", timeit.timeit(code_dict_comp_ternary, setup=setup, number=100000))
print("Dict Comp (or):", timeit.timeit(code_dict_comp_or, setup=setup, number=100000))
