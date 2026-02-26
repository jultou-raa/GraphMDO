import os
from falkordb import FalkorDB


class FalkorDBClient:
    """
    Singleton class to manage the connection to the FalkorDB database.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalkorDBClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initializes the FalkorDB connection.
        """
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", 6379))
        self.client = FalkorDB(host=host, port=port)
        self.graph_name = "mdo_graph"
        self.graph = self.client.select_graph(self.graph_name)

    def get_graph(self):
        """
        Returns the graph object for executing queries.
        """
        return self.graph

    def close(self):
        """
        Closes the connection.
        """
        # FalkorDB client handles connection pooling, but explicit close might be needed
        # depending on the implementation. For now, we rely on the client's management.
        pass
