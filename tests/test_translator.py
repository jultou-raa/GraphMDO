"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest
from unittest.mock import patch

import numpy as np

from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.schema import (
    FixedParam,
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolSpec,
)


def coupled_schema(**overrides) -> StudySchema:
    """x -> A -> y -> B -> c -> A: ``y = x + 0.5 c`` and ``c = 0.5 y``."""
    variables = {
        "x": RangeVar(name="x", lower=0.0, upper=10.0, initial=2.0),
        "y": StateVar(name="y"),
        "c": StateVar(name="c", initial_guess=0.5),
    }
    variables.update(overrides)
    return StudySchema(
        variables=list(variables.values()),
        tools=[
            ToolSpec(name="A", inputs=["x", "c"], outputs=["y"]),
            ToolSpec(name="B", inputs=["y"], outputs=["c"]),
        ],
    )


COUPLED_REGISTRY = {
    "A": lambda x, c: x + 0.5 * c,
    "B": lambda y: 0.5 * y,
}


class TestTranslator(unittest.TestCase):
    def test_build_problem_invalid_tool(self):
        schema = StudySchema(tools=[ToolSpec(name="InvalidTool", fidelity="high")])

        builder = GraphProblemBuilder(schema)

        tool_registry = {}  # Empty registry

        with self.assertRaises(ValueError) as context:
            builder.build_problem(tool_registry)
        self.assertIsInstance(context.exception, StudyValidationError)
        self.assertEqual(
            [f.code for f in context.exception.report.errors], ["UNREGISTERED_TOOL"]
        )
        self.assertIn("InvalidTool", str(context.exception))

    def test_build_problem_success(self):
        schema = StudySchema(
            variables=[FixedParam(name="x", value=1.0), StateVar(name="y")],
            tools=[ToolSpec(name="ToolA", inputs=["x"], outputs=["y"])],
        )

        builder = GraphProblemBuilder(schema)

        def tool_func(x):
            return x

        tool_registry = {"ToolA": tool_func}

        mda = builder.build_problem(tool_registry)

        self.assertEqual(mda.name, "MDAChain")
        self.assertIs(builder.schema, schema)

        # Fixed values become the defaults of the MDA inputs
        np.testing.assert_array_equal(mda.default_input_data["x"], np.array([1.0]))

    def test_built_problem_executes_tool(self):
        schema = StudySchema(
            variables=[
                FixedParam(name="x", value=3.0),
                FixedParam(name="y", value=-4.0),
                StateVar(name="f_xy"),
            ],
            tools=[
                ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy"]),
            ],
        )

        def paraboloid(x, y):
            return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        mda = GraphProblemBuilder(schema).build_problem({"Paraboloid": paraboloid})
        out = mda.execute({"x": np.array([3.0]), "y": np.array([-4.0])})

        self.assertEqual(mda.disciplines[0].name, "Paraboloid")
        self.assertAlmostEqual(float(np.asarray(out["f_xy"]).flat[0]), -15.0)

    def test_fixed_parameter_without_design_variable_runs_with_its_default(self):
        schema = StudySchema(
            variables=[FixedParam(name="rho", value=2.0), StateVar(name="f")],
            tools=[ToolSpec(name="T", inputs=["rho"], outputs=["f"])],
        )

        mda = GraphProblemBuilder(schema).build_problem({"T": lambda rho: 3 * rho})
        out = mda.execute()

        self.assertEqual(float(np.asarray(out["f"]).flat[0]), 6.0)


class TestMdaDefaults(unittest.TestCase):
    def test_coupling_initial_guess_is_a_default_and_seeds_the_solve(self):
        schema = coupled_schema(y=StateVar(name="y", initial_guess=1.0))
        mda = GraphProblemBuilder(schema).build_problem(COUPLED_REGISTRY)

        np.testing.assert_array_equal(mda.default_input_data["c"], np.array([0.5]))
        np.testing.assert_array_equal(mda.default_input_data["y"], np.array([1.0]))
        out = mda.execute({"x": np.array([3.0])})
        # y = x + 0.5 c and c = 0.5 y  ->  y = 4, c = 2
        self.assertAlmostEqual(float(np.asarray(out["y"]).flat[0]), 4.0, places=3)
        self.assertAlmostEqual(float(np.asarray(out["c"]).flat[0]), 2.0, places=3)

    def test_vector_initial_guess_keeps_its_shape(self):
        schema = coupled_schema(c=StateVar(name="c", initial_guess=[1.0, 2.0]))

        mda = GraphProblemBuilder(schema).build_problem(COUPLED_REGISTRY)

        np.testing.assert_array_equal(mda.default_input_data["c"], np.array([1.0, 2.0]))

    def test_state_without_initial_guess_has_no_default(self):
        schema = coupled_schema(c=StateVar(name="c"))

        mda = GraphProblemBuilder(schema).build_problem(COUPLED_REGISTRY)

        self.assertNotIn("c", mda.default_input_data)

    def test_design_variable_initial_is_not_a_default(self):
        mda = GraphProblemBuilder(coupled_schema()).build_problem(COUPLED_REGISTRY)

        self.assertIn("x", mda.input_grammar)
        self.assertNotIn("x", mda.default_input_data)

    def test_fixed_parameter_value_is_a_default_next_to_a_design_variable(self):
        schema = StudySchema(
            variables=[
                RangeVar(name="v", lower=0.0, upper=1.0, initial=0.25),
                FixedParam(name="rho", value=1.225),
                StateVar(name="f"),
            ],
            tools=[ToolSpec(name="T", inputs=["v", "rho"], outputs=["f"])],
        )

        mda = GraphProblemBuilder(schema).build_problem({"T": lambda v, rho: v * rho})

        np.testing.assert_array_equal(mda.default_input_data["rho"], np.array([1.225]))
        self.assertNotIn("v", mda.default_input_data)


class TestRegistryErrors(unittest.TestCase):
    def test_every_bad_tool_is_listed_in_one_error(self):
        schema = StudySchema(
            variables=[
                RangeVar(name="x", lower=0.0, upper=1.0),
                StateVar(name="a"),
                StateVar(name="b"),
                StateVar(name="c"),
                StateVar(name="d"),
            ],
            tools=[
                ToolSpec(name="Good", inputs=["x"], outputs=["a"]),
                ToolSpec(name="Missing1", inputs=["x"], outputs=["b"]),
                ToolSpec(name="BadSignature", inputs=["x"], outputs=["c"]),
                ToolSpec(name="Missing2", inputs=["x"], outputs=["d"]),
            ],
        )
        registry = {
            "Good": lambda x: x,
            "BadSignature": lambda other: other,
        }

        with self.assertRaises(StudyValidationError) as context:
            GraphProblemBuilder(schema).build_problem(registry)

        findings = [
            (finding.code, finding.names[0])
            for finding in context.exception.report.errors
        ]
        self.assertEqual(
            findings,
            [
                ("UNREGISTERED_TOOL", "Missing1"),
                ("UNREGISTERED_TOOL", "Missing2"),
                ("SIGNATURE_MISMATCH", "BadSignature"),
            ],
        )

    def test_signature_warnings_do_not_block_the_build(self):
        schema = StudySchema(
            variables=[FixedParam(name="x", value=1.0), StateVar(name="y")],
            tools=[ToolSpec(name="T", inputs=["x"], outputs=["y"])],
        )

        mda = GraphProblemBuilder(schema).build_problem(
            {"T": lambda x, scale=2.0: x * scale}
        )

        self.assertEqual(mda.name, "MDAChain")


class TestConsistencyCheck(unittest.TestCase):
    def test_disciplines_are_checked_before_the_mda_is_created(self):
        with patch(
            "mdo_framework.core.translator.check_disciplines_consistency"
        ) as check:
            GraphProblemBuilder(coupled_schema()).build_problem(COUPLED_REGISTRY)

        check.assert_called_once()
        disciplines, log_message, raise_error = check.call_args.args
        self.assertEqual([d.name for d in disciplines], ["A", "B"])
        self.assertFalse(log_message)
        self.assertTrue(raise_error)

    def test_two_tools_producing_the_same_output_are_rejected(self):
        # The schema model forbids this; build it unvalidated to prove the
        # builder still refuses it (defence in depth).
        schema = StudySchema.model_construct(
            variables=[
                RangeVar(name="x", lower=0.0, upper=1.0),
                StateVar(name="y"),
            ],
            tools=[
                ToolSpec(name="First", inputs=["x"], outputs=["y"]),
                ToolSpec(name="Second", inputs=["x"], outputs=["y"]),
            ],
        )

        with self.assertRaisesRegex(ValueError, "y"):
            GraphProblemBuilder(schema).build_problem(
                {"First": lambda x: x, "Second": lambda x: 2 * x}
            )


if __name__ == "__main__":
    unittest.main()
