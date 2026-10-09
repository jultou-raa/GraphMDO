# Bayesian-optimization layer

`BayesianOptimizer` runs one of these GEMSEO libraries. `BaseBOLibrary` owns everything that must not depend on the backend; a backend only implements `_setup`, `_ask` and `_tell`.

| Algorithm | Library | Backend |
| --- | --- | --- |
| `Ax_Bayesian` (default) | `AxOptimizationLibrary` | Ax: Sobol for `n_init` trials, then BoTorch. |
| `BO_RandomSearch` | `RandomSearchLibrary` | Seeded uniform sampling (log-uniform for `scaling="log"`), the reference backend of the tests. |

# Driver

::: mdo_framework.optimization.bo_library

# Types

::: mdo_framework.optimization.bo_types

# Random search

::: mdo_framework.optimization.random_search

# Ax backend

::: mdo_framework.optimization.ax_algo_lib

# Errors

::: mdo_framework.optimization.errors
