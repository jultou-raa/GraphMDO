with open("tests/test_optimizer.py", "r") as f:
    code = f.read()

# Fix `LocalEvaluator` missing output: `Missing required names: g_xy`
# In `test_local_evaluator_scalar` and `iterable`:
# I only updated `f_xy` in the mock `_run`, but `MockDisc` output grammar expects `g_xy` as well!
code = code.replace(
"""        def dummy_run(self, input_data):
            self.local_data["f_xy"] = np.array([1.23])""",
"""        def dummy_run(self, input_data):
            self.local_data["f_xy"] = np.array([1.23])
            self.local_data["g_xy"] = np.array([0.0])"""
)

code = code.replace(
"""        def dummy_run2(self, input_data):
            self.local_data["f_xy"] = np.array([4.56, 1.0])""",
"""        def dummy_run2(self, input_data):
            self.local_data["f_xy"] = np.array([4.56, 1.0])
            self.local_data["g_xy"] = np.array([0.0])"""
)

with open("tests/test_optimizer.py", "w") as f:
    f.write(code)
