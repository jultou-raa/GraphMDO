"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fakes.falkordb import FakeGraph

from mdo_framework.db.graph_manager import GraphManager


def _load_main_module():
    """Load the top-level main.py script as a module without installing it as a package."""
    main_path = Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("main", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main"] = module
    spec.loader.exec_module(module)
    return module


main_module = _load_main_module()


class TestMainIntegration(unittest.TestCase):
    def test_main_builds_the_typed_paraboloid_and_optimizes_it(self):
        """main() runs the whole demo on a graph that only accepts typed models."""
        graph = FakeGraph()
        output = io.StringIO()

        with (
            patch.object(
                main_module, "GraphManager", lambda: GraphManager(graph=graph)
            ),
            redirect_stdout(output),
        ):
            main_module.main()

        text = output.getvalue()
        self.assertIn("Paraboloid Inputs: ['x', 'y']", text)
        self.assertIn("Optimization Complete.", text)
        self.assertNotIn("failed", text)

        schema = GraphManager(graph=graph).get_study_schema()
        self.assertEqual(
            [(variable.kind, variable.name) for variable in schema.variables],
            [("range", "x"), ("range", "y"), ("state", "f_xy"), ("state", "c_xy")],
        )
        self.assertEqual(
            [(tool.name, tool.inputs, tool.outputs) for tool in schema.tools],
            [("Paraboloid", ["x", "y"], ["f_xy", "c_xy"])],
        )
