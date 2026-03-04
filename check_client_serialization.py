import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['openmdao'] = MagicMock()
sys.modules['openmdao.api'] = MagicMock()
sys.modules['httpx'] = MagicMock()
sys.modules['torch'] = MagicMock()

# We need real ax if possible to check attributes, but since uv sync failed...
# Let's try to see if we can import just ax.api.client without full botorch/torch if they are not strictly needed for the class definition.
try:
    from ax.api.client import Client
    client = Client()
    print("Methods of Client:", [m for m in dir(client) if not m.startswith('_')])
except Exception as e:
    print(f"Error: {e}")
