# BayesianOptimizer

The optimizer API supports:

- Local and remote evaluation backends.
- Explicit inequality constraints using `<=` and `>=`.
- Trial-history exports as explicit records with `parameters` and `objectives`.
- Typed failures for configuration, transport, contract, and execution errors.

::: mdo_framework.optimization.optimizer.BayesianOptimizer

# Evaluator

::: mdo_framework.optimization.optimizer.Evaluator

# LocalEvaluator

::: mdo_framework.core.evaluators.LocalEvaluator

# RemoteEvaluator

`RemoteEvaluator` distinguishes transport failures from invalid execution-service responses so service layers can map them to different HTTP statuses.

::: mdo_framework.optimization.optimizer.RemoteEvaluator
