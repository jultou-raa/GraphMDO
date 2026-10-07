"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest

from mdo_framework.core.topology import (
    ResolvedInputs,
    TopologicalAnalyzer,
    build_variable_specs,
    to_parameter_definition,
)
from mdo_framework.schema import (
    ChoiceVar,
    FixedParam,
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolSpec,
)

X = RangeVar(name="x", lower=0.0, upper=10.0)
Y = ChoiceVar(name="y", choices=["A", "B"])


def codes(error: StudyValidationError) -> list[str]:
    return [finding.code for finding in error.report.errors]


class TestTopologicalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.schema = StudySchema(
            variables=[
                X,
                Y,
                StateVar(name="z"),
                RangeVar(name="unused_in", lower=0.0, upper=1.0),
                StateVar(name="out1"),
                StateVar(name="out2"),
                StateVar(name="out3"),
            ],
            tools=[
                ToolSpec(name="Tool1", inputs=["x", "y"], outputs=["z"]),
                ToolSpec(name="Tool2", inputs=["z"], outputs=["out1", "out2"]),
                ToolSpec(name="UnusedTool", inputs=["unused_in"], outputs=["out3"]),
            ],
        )

    def test_resolve_dependencies_full(self):
        resolved = TopologicalAnalyzer(self.schema).resolve_dependencies(["out1"])

        # Unused inputs/tools should not be here
        self.assertEqual(
            resolved,
            ResolvedInputs(
                design_variables=(X, Y),
                fixed_parameters=(),
                tools=("Tool1", "Tool2"),
            ),
        )

    def test_extract_parameters(self):
        analyzer = TopologicalAnalyzer(self.schema)
        resolved = analyzer.resolve_dependencies(["out1"])

        self.assertEqual(
            analyzer.extract_parameters(resolved.design_variables),
            [
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.0, 10.0],
                    "value_type": "float",
                },
                {
                    "name": "y",
                    "type": "choice",
                    "values": ["A", "B"],
                    "value_type": "str",
                },
            ],
        )

    def test_resolve_dependencies_missing_var(self):
        analyzer = TopologicalAnalyzer(self.schema)
        with self.assertRaises(StudyValidationError) as context:
            analyzer.resolve_dependencies(["missing_out"])
        self.assertEqual(codes(context.exception), ["UNKNOWN_OUTPUT"])
        self.assertIn("missing_out", str(context.exception))

    def test_target_that_no_tool_produces_is_reported(self):
        analyzer = TopologicalAnalyzer(self.schema)
        with self.assertRaises(StudyValidationError) as context:
            analyzer.resolve_dependencies(["x"])
        self.assertEqual(codes(context.exception), ["NOT_PRODUCED"])

    def test_every_problem_is_reported_in_one_error(self):
        schema = StudySchema(
            variables=[X, StateVar(name="orphan"), StateVar(name="f")],
            tools=[ToolSpec(name="T", inputs=["x", "orphan"], outputs=["f"])],
        )
        with self.assertRaises(StudyValidationError) as context:
            TopologicalAnalyzer(schema).resolve_dependencies(["f", "ghost", "x"])
        self.assertEqual(
            codes(context.exception),
            ["UNKNOWN_OUTPUT", "NOT_PRODUCED", "UNPRODUCED_STATE"],
        )
        message = str(context.exception)
        for name in ("ghost", "'x'", "orphan"):
            self.assertIn(name, message)


class TestFixedParametersAreNotDesignVariables(unittest.TestCase):
    """#38: a valued input without bounds must never be optimized."""

    def test_fixed_inputs_are_separated_from_design_variables(self):
        rho = FixedParam(name="rho", value=1.225)
        gravity = FixedParam(name="g", value=9.81)
        speed = RangeVar(name="v", lower=0.0, upper=10.0)
        schema = StudySchema(
            variables=[speed, rho, gravity, StateVar(name="f")],
            tools=[ToolSpec(name="T", inputs=["v", "rho", "g"], outputs=["f"])],
        )
        analyzer = TopologicalAnalyzer(schema)

        resolved = analyzer.resolve_dependencies(["f"])

        self.assertEqual(resolved.design_variables, (speed,))
        self.assertEqual(resolved.fixed_parameters, (rho, gravity))
        self.assertEqual(
            [p["name"] for p in analyzer.extract_parameters(resolved.design_variables)],
            ["v"],
        )

    def test_only_fixed_inputs_leave_no_design_variable(self):
        schema = StudySchema(
            variables=[FixedParam(name="rho", value=1.0), StateVar(name="f")],
            tools=[ToolSpec(name="T", inputs=["rho"], outputs=["f"])],
        )

        resolved = TopologicalAnalyzer(schema).resolve_dependencies(["f"])

        self.assertEqual(resolved.design_variables, ())
        self.assertEqual([p.name for p in resolved.fixed_parameters], ["rho"])


class TestDesignVariableOrder(unittest.TestCase):
    def test_order_follows_the_schema_not_tool_inputs_or_names(self):
        schema = StudySchema(
            variables=[
                RangeVar(name="zeta", lower=0.0, upper=1.0),
                ChoiceVar(name="mid", choices=[1, 2]),
                RangeVar(name="alpha", lower=0.0, upper=1.0),
                StateVar(name="f"),
            ],
            tools=[ToolSpec(name="T", inputs=["alpha", "mid", "zeta"], outputs=["f"])],
        )
        analyzer = TopologicalAnalyzer(schema)

        resolved = analyzer.resolve_dependencies(["f"])

        self.assertEqual(
            [variable.name for variable in resolved.design_variables],
            ["zeta", "mid", "alpha"],
        )
        self.assertEqual(
            [p["name"] for p in analyzer.extract_parameters(resolved.design_variables)],
            ["zeta", "mid", "alpha"],
        )


class TestParameterDefinitions(unittest.TestCase):
    def test_range_definition(self):
        self.assertEqual(
            to_parameter_definition(RangeVar(name="n", lower=1, upper=5)),
            {
                "name": "n",
                "type": "range",
                "bounds": [1.0, 5.0],
                "value_type": "float",
            },
        )
        integer = to_parameter_definition(
            RangeVar(name="n", lower=1, upper=5, value_type="int")
        )
        self.assertEqual(integer["value_type"], "int")
        self.assertEqual(integer["bounds"], [1, 5])

    def test_choice_definition_keeps_the_declared_value_type(self):
        for choices, value_type in (
            (["a", "b"], "str"),
            ([10, 20, 30], "int"),
            ([0.5, 2.0], "float"),
            ([True, False], "bool"),
        ):
            definition = to_parameter_definition(ChoiceVar(name="c", choices=choices))
            self.assertEqual(
                definition,
                {
                    "name": "c",
                    "type": "choice",
                    "values": choices,
                    "value_type": value_type,
                },
            )

    def test_variable_specs_cover_only_design_variables(self):
        schema = StudySchema(
            variables=[
                X,
                Y,
                FixedParam(name="rho", value=1.0),
                StateVar(name="f"),
            ],
            tools=[ToolSpec(name="T", inputs=["x", "y", "rho"], outputs=["f"])],
        )

        specs = build_variable_specs(schema)

        self.assertEqual(list(specs), ["x", "y"])
        self.assertEqual(specs["x"], to_parameter_definition(X))
        self.assertEqual(specs["y"], to_parameter_definition(Y))


if __name__ == "__main__":
    unittest.main()
