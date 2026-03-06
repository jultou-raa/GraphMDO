from pathlib import Path

test_file = Path("tests/test_services.py")
text = test_file.read_text()

# Append a specific test for the missing objective
new_test = """
    def test_execute_problem_missing_objective(self):
        from services.execution.main import execute_problem
        from unittest.mock import MagicMock

        mock_prob = MagicMock()
        mock_prob.execute.return_value = {"known_obj": 1.0}

        # 'missing_obj' should trigger the `if val is None: results[obj] = 0.0` logic
        results = execute_problem(mock_prob, inputs={"x": 1.0}, objectives=["known_obj", "missing_obj"])
        self.assertEqual(results["missing_obj"], 0.0)

"""

# Insert it at the end of TestExecutionService class
text = text.replace("    def test_evaluate_transformation_failure(self):", new_test + "    def test_evaluate_transformation_failure(self):")
test_file.write_text(text)
print("Test added")
