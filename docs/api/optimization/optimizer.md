# BayesianOptimizer

The optimizer API supports:

- Local and remote evaluation backends.
- Typed `ObjectiveSpec` and `ConstraintSpec` inputs (`<=` and `>=`, with an optional per-constraint tolerance) and the design variables resolved from the study schema.
- A result with the best point, its feasibility and constraint margins, the Pareto front of multi-objective runs, a trial history with one record per trial (`phase`, `status`, `parameters`, `objectives`, `constraints`), the `stop_reason` and the evaluation counts.
- Typed failures for configuration, transport, contract, and execution errors; the execution errors carry the `partial_result` of the run.

::: mdo_framework.optimization.optimizer.BayesianOptimizer

# Evaluator

::: mdo_framework.optimization.optimizer.Evaluator

# LocalEvaluator

::: mdo_framework.core.evaluators.LocalEvaluator

# RemoteEvaluator

`RemoteEvaluator` distinguishes transport failures from invalid execution-service responses so service layers can map them to different HTTP statuses.

::: mdo_framework.optimization.optimizer.RemoteEvaluator
