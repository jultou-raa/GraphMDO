# GEMSEO Library Utilization Assessment in GraphMDO

## Executive Summary
This report evaluates the integration and utilization of the `gemseo` (Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization) library within the GraphMDO repository. The analysis maps the current implementation against the complete suite of `gemseo` features, highlighting missed opportunities and redundant custom logic, while providing actionable recommendations to maximize the library's potential.

## 1. Current Usage Profile
A code-wide analysis reveals that `gemseo` is utilized as the core execution engine for multidisciplinary design analysis (MDA). Specifically:

*   **Disciplines (`gemseo.core.discipline.Discipline`):** GraphMDO uses `gemseo`'s foundational `Discipline` class to construct `ToolComponent` wrappers (`src/mdo_framework/core/components.py`). This allows standard Python functions to be recognized as mathematical nodes within an MDO graph.
*   **Multidisciplinary Design Analysis (`gemseo.mda.factory.MDAFactory`):** GraphMDO leverages the `MDAFactory` to instantiate an `MDAChain` (`src/mdo_framework/core/translator.py`). This handles the execution of sequentially coupled disciplines and manages cyclical data dependencies using internal Gauss-Seidel iterations.
*   **Execution Paradigm:** The system relies on local data passing (`local_data` with NumPy arrays) and executes the MDA through a custom `LocalEvaluator` class (`src/mdo_framework/core/evaluators.py`).

## 2. Untapped Potential & Identified Gaps
While the baseline execution relies on `gemseo`, GraphMDO significantly underutilizes the library's higher-level orchestration, optimization, and analytic capabilities.

### 2.1 Optimization & MDO Formulations
*   **Gap:** GraphMDO implements a bespoke `BayesianOptimizer` utilizing the `ax-platform` and `botorch` (`src/mdo_framework/optimization/optimizer.py`), driving optimization externally via HTTP microservices or local evaluation loops.
*   **Missed `gemseo` Feature:** `gemseo` provides native `MDOScenario` classes (e.g., MDF, IDF, SAND formulations) that directly manage the optimization problem, handle constraints natively via unified design spaces, and offer a wide array of optimization algorithms (including gradient-based and derivative-free solvers). The IDF formulation in `gemseo` natively supports parallel process scaling (`n_processes`), a feature currently absent in GraphMDO's synchronous evaluation loop.

### 2.2 Surrogate Modeling
*   **Gap:** GraphMDO uses a custom wrapper around the `smt` library (`SMTSurrogate` in `src/mdo_framework/core/surrogates.py`) to build Kriging (KRG) and KPLS surrogate models.
*   **Missed `gemseo` Feature:** `gemseo` natively supports surrogate model integration (based on `scikit-learn` and `OpenTURNS`). By using `gemseo` surrogates, GraphMDO could seamlessly replace expensive analytical disciplines with machine-learning-backed approximations within an `MDOScenario` without requiring custom bridging code.

### 2.3 Design of Experiments (DOE)
*   **Gap:** GraphMDO relies on `ax-platform` (specifically Sobol generation via `Models.SOBOL`) to perform the initial quasi-random exploration of the design space.
*   **Missed `gemseo` Feature:** `gemseo` contains a dedicated `DOEScenario` capability with extensive algorithm support (Uniform, Latin Hypercube, Full Factorial, etc.). `DOEScenario` features built-in parallelization (`n_processes > 1`) for rapid concurrent evaluations of observables.

### 2.4 Post-Processing and Analytics
*   **Gap:** Post-processing is handled manually. Results are extracted via dictionary manipulation and converted to standard types for JSON serialization by the optimization service.
*   **Missed `gemseo` Feature:** `gemseo` boasts extensive visualization and post-processing tools (e.g., OptHistoryView, XDSM graph generation). Recent versions of `gemseo` allow configuration of these post-processors securely via Pydantic models.

## 3. Actionable Recommendations

To fully harness the power of `gemseo`, improve maintainability, and reduce external dependencies, GraphMDO should consider the following strategic refactoring steps:

1.  **Migrate to `MDOScenario`:** Replace the custom external optimization loop (currently relying on `ax-platform`) with `gemseo`'s native `MDOScenario`. If Bayesian Optimization is strictly required, explore registering a custom Bayesian algorithm within the `gemseo` optimization factory, rather than maintaining a completely separate framework.
2.  **Adopt Native Surrogate Models:** Deprecate the `SMTSurrogate` wrapper in favor of `gemseo`'s native surrogate modeling capabilities. This will standardise the interface and allow surrogates to act natively as `gemseo` disciplines within an `MDAChain`.
3.  **Implement `DOEScenario` for Exploration:** Utilize `DOEScenario` for design space exploration. Leverage its `n_processes` argument to execute independent design evaluations concurrently across multiple CPU cores, drastically reducing initialization time.
4.  **Leverage XDSM and Post-Processing:** Integrate `scenario.xdsmize()` to visually validate complex MDO graph topologies. Use `scenario.post_process()` configured via Pydantic settings to generate rich, standardized optimization histories and convergence plots without writing custom visualization scripts.
