with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

seed_code = """    @staticmethod
    def _extract_seed_params(x_seed: np.ndarray, design_space: Any) -> dict[str, float]:
        seed_params: dict[str, float] = {}
        seed_offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            for j in range(size):
                param_name = _get_param_name(var_name, j, size)
                seed_params[param_name] = float(x_seed[seed_offset + j])
            seed_offset += size
        return seed_params

    @staticmethod
    def _extract_seed_results(output: dict[str, Any], metric_names: set[str], c_names: set[str]) -> dict[str, float]:
        seed_results = {}
        for metric in metric_names:
            val = output.get(metric)
            if val is not None:
                if isinstance(val, np.ndarray) and val.size > 1:
                    if metric in c_names:
                        logger.warning(
                            "Multi-dimensional array detected for constraint '%s'. "
                            "Using np.max for aggregation, which may create a non-smooth gradient landscape. "
                            "Consider defining a smooth aggregation function natively within the discipline.",
                            metric
                        )
                    seed_results[metric] = float(np.max(val))
                else:
                    seed_results[metric] = float(val) if not isinstance(val, np.ndarray) else float(val[0])
        return seed_results

    def _seed_database(
        self, client: Client, problem: OptimizationProblem, design_space: Any
    ) -> None:
        obj_names = problem.objective.name
        if not isinstance(obj_names, list):
            obj_names = [obj_names]

        c_names = {c.name for c in problem.constraints}
        metric_names = set(obj_names) | c_names

        for i, (x_hash, output) in enumerate(problem.database.items()):
            x_seed = x_hash.unwrap()
            seed_params = self._extract_seed_params(x_seed, design_space)
            seed_results = self._extract_seed_results(output, metric_names, c_names)

            if seed_results:
                if i == 0:
                    trial_idx = client.attach_baseline(parameters=seed_params)
                else:
                    trial_idx = client.attach_trial(parameters=seed_params)
                client.complete_trial(trial_index=trial_idx, raw_data=seed_results)"""

start_idx = content.find("    def _seed_database(")
end_idx = content.find("    def _configure_client(", start_idx)

content = content[:start_idx] + seed_code + "\n\n" + content[end_idx:]

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
