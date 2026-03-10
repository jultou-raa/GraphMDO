"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from mdo_framework.optimization.parameter_codec import (
    ParameterDefinitionError,
    ParameterValueError,
    build_parameter_lookup,
    coerce_scalar,
    decode_parameter_value,
    encode_parameter_value,
)


class TestParameterCodec(unittest.TestCase):
    def test_round_trip_variants(self):
        cases = [
            ({"name": "x", "type": "range", "value_type": "float"}, 1.25, 1.25),
            ({"name": "n", "type": "range", "value_type": "int"}, 1.6, 2),
            ({"name": "c_str", "type": "choice", "values": ["A", "B"]}, "B", "B"),
            (
                {"name": "c_bool", "type": "choice", "values": [False, True]},
                True,
                True,
            ),
            ({"name": "c_single", "type": "choice", "values": [42]}, 42, 42),
        ]

        for parameter, input_value, expected_value in cases:
            with self.subTest(parameter=parameter["name"], input_value=input_value):
                encoded_value = encode_parameter_value(parameter, input_value)
                decoded_value = decode_parameter_value(parameter, encoded_value)
                self.assertEqual(decoded_value, expected_value)

    def test_helper_functions(self):
        scalar_cases = [
            (np.array([1.5]), 1.5),
            (np.array([1.0, 2.0]), [1.0, 2.0]),
            (np.float64(2.5), 2.5),
        ]

        for raw_value, expected_value in scalar_cases:
            with self.subTest(raw_value=raw_value):
                self.assertEqual(coerce_scalar(raw_value), expected_value)

        self.assertEqual(build_parameter_lookup(None), {})
        self.assertEqual(
            decode_parameter_value({"name": "label", "type": "range"}, "A"),
            "A",
        )

    def test_rejects_invalid_choice_definition(self):
        parameter = {"name": "c_bad", "type": "choice", "values": []}
        failure_cases = [
            ("encode", lambda: encode_parameter_value(parameter, 0)),
            ("decode", lambda: decode_parameter_value(parameter, 0)),
        ]

        for label, action in failure_cases:
            with self.subTest(action=label):
                with self.assertRaises(ParameterDefinitionError):
                    action()

    def test_rejects_invalid_choice_values(self):
        failure_cases = [
            (
                "out_of_bounds_index",
                lambda: decode_parameter_value(
                    {"name": "c_str", "type": "choice", "values": ["A", "B"]},
                    3,
                ),
            ),
            (
                "non_numeric_decode",
                lambda: decode_parameter_value(
                    {"name": "c_str", "type": "choice", "values": ["A", "B"]},
                    "invalid-index",
                ),
            ),
            (
                "bool_to_int_choice",
                lambda: encode_parameter_value(
                    {"name": "c_int", "type": "choice", "values": [0, 1]},
                    True,
                ),
            ),
        ]

        for label, action in failure_cases:
            with self.subTest(case=label):
                with self.assertRaises(ParameterValueError):
                    action()
