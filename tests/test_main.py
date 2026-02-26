import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


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
    @patch("mdo_framework.db.graph_manager.GraphManager")
    def test_main_execution(self, mock_gm_cls):
        """Verify main() runs without raising, using a mocked GraphManager."""
        mock_gm = MagicMock()
        mock_gm_cls.return_value = mock_gm
        mock_gm.get_graph_schema.return_value = {
            "tools": [
                {
                    "name": "Paraboloid",
                    "fidelity": "high",
                    "inputs": ["x", "y"],
                    "outputs": ["f_xy"],
                }
            ],
            "variables": [
                {"name": "x", "value": 0.0},
                {"name": "y", "value": 0.0},
                {"name": "f_xy"},
            ],
        }

        with patch.object(main_module, "GraphManager", mock_gm_cls):
            try:
                main_module.main()
            except Exception as e:
                self.fail(f"main.py raised an Exception unexpectedly: {e}")
