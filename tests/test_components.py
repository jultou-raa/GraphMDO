"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from mdo_framework.core.components import ToolComponent


def simple_func(x, y):
    return x + y


class TestToolComponent(unittest.TestCase):
    def test_simple_execution(self):
        comp = ToolComponent(
            name="add",
            func=simple_func,
            inputs=["x", "y"],
            outputs=["z"],
        )

        input_data = {"x": np.array([2.0]), "y": np.array([3.0])}
        out = comp.execute(input_data)
        self.assertAlmostEqual(out["z"][0], 5.0)

    def test_dict_output(self):
        def dict_func(a):
            return {"b": a * 2, "c": a + 1}

        comp = ToolComponent(
            name="dict_comp",
            func=dict_func,
            inputs=["a"],
            outputs=["b", "c"],
        )

        input_data = {"a": np.array([10.0])}
        out = comp.execute(input_data)
        self.assertAlmostEqual(out["b"][0], 20.0)
        self.assertAlmostEqual(out["c"][0], 11.0)

    def test_tuple_output(self):
        def tuple_func(a):
            return a * 2, a + 1

        comp = ToolComponent(
            name="tuple_comp",
            func=tuple_func,
            inputs=["a"],
            outputs=["b", "c"],
        )

        input_data = {"a": np.array([10.0])}
        out = comp.execute(input_data)
        self.assertAlmostEqual(out["b"][0], 20.0)
        self.assertAlmostEqual(out["c"][0], 11.0)

    def test_derivatives_setup(self):
        comp = ToolComponent(
            name="deriv_comp",
            func=simple_func,
            inputs=["x", "y"],
            outputs=["z"],
            derivatives=True,
        )
        # In GEMSEO, analytical derivatives are enabled or computed via jacobian methods.
        input_data = {"x": np.array([2.0]), "y": np.array([3.0])}
        comp.execute(input_data)
        # Since _compute_jacobian is empty, we just verify it runs without crashing.
        comp._compute_jacobian(inputs=["x", "y"], outputs=["z"])

    def test_kwargs_mapping_uses_input_names(self):
        """Function parameter names must match the declared input names.

        The previous 'positional fallback' was removed because it silently
        swapped variables when the input order differed from the function
        signature. Explicit kwargs mapping is now enforced.
        """

        def matching_args(x, y):
            return x + y

        comp = ToolComponent(
            name="matching",
            func=matching_args,
            inputs=["x", "y"],
            outputs=["z"],
        )

        input_data = {"x": np.array([1.0]), "y": np.array([2.0])}
        out = comp.execute(input_data)
        self.assertAlmostEqual(out["z"][0], 3.0)

    def test_mismatched_parameter_names_raise(self):
        """A function whose parameter names don't match input names must raise."""

        def mismatch_args(a, b):
            return a + b

        comp = ToolComponent(
            name="mismatch",
            func=mismatch_args,
            inputs=["x", "y"],
            outputs=["z"],
        )

        input_data = {"x": np.array([1.0]), "y": np.array([2.0])}
        with self.assertRaises(TypeError):
            comp.execute(input_data)


if __name__ == "__main__":
    unittest.main()
