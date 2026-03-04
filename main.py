from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.db.graph_manager import GraphManager
from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.core.evaluators import LocalEvaluator


# --- Sellar Problem Functions ---
def paraboloid_func(x, y):
    """
    f(x, y) = (x-3)**2 + xy + (y+4)**2 - 3
    """
    f_xy = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0
    c_xy = x - y
    return {"f_xy": f_xy, "c_xy": c_xy}


# --- Tool Registry ---
TOOL_REGISTRY = {"Paraboloid": paraboloid_func}


def main():
    # 1. Initialize Graph Manager
    gm = GraphManager()

    # Clean previous run
    try:
        gm.clear_graph()
    except Exception as e:
        print(f"Warning: Could not clear graph (maybe DB not ready?): {e}")

    print("Building Graph in FalkorDB...")

    # 2. Populate Graph (Paraboloid Problem)
    # Variables
    gm.add_variable("x", value=0.0, lower=-10.0, upper=10.0)
    gm.add_variable("y", value=0.0, lower=-10.0, upper=10.0)
    gm.add_variable("f_xy", value=0.0)
    gm.add_variable("c_xy", value=0.0)

    # Tool
    gm.add_tool("Paraboloid")

    # Connections
    # Input -> Tool
    gm.connect_input_to_tool("x", "Paraboloid")
    gm.connect_input_to_tool("y", "Paraboloid")

    # Tool -> Output
    gm.connect_tool_to_output("Paraboloid", "f_xy")
    gm.connect_tool_to_output("Paraboloid", "c_xy")

    # Verify Topology
    inputs = gm.get_tool_inputs("Paraboloid")
    print(f"Paraboloid Inputs: {inputs}")

    # 3. Translate to OpenMDAO
    print("Translating Graph to OpenMDAO Problem...")
    builder = GraphProblemBuilder(gm.get_graph_schema())

    try:
        prob = builder.build_problem(TOOL_REGISTRY)
    except Exception as e:
        print(f"Translation failed: {e}")
        return

    # 4. Optimization
    print("Starting Optimization...")

    from mdo_framework.core.topology import TopologicalAnalyzer

    schema = gm.get_graph_schema()

    analyzer = TopologicalAnalyzer(schema)
    design_vars, _ = analyzer.resolve_dependencies(["f_xy"])
    parameters = analyzer.extract_parameters(design_vars)

    evaluator = LocalEvaluator(prob)

    optimizer = BayesianOptimizer(
        evaluator=evaluator,
        parameters=parameters,
        objectives=[{"name": "f_xy", "minimize": True}],
    )

    try:
        result = optimizer.optimize(n_steps=10, n_init=5)
        print("Optimization Complete.")
        print("Best Result:", result["best_parameters"])
        print("Best Objectives:", result["best_objectives"])
    except Exception as e:
        print(f"Optimization failed: {e}")


if __name__ == "__main__":
    main()
