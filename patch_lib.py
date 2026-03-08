with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

# Update constructor to take config_factory
content = content.replace("""    def __init__(self, algo_name: str = "Ax_Bayesian", client_factory=None) -> None:
        super().__init__(algo_name=algo_name)
        self.client_factory = client_factory or Client""", """    def __init__(self, algo_name: str = "Ax_Bayesian", client_factory=None, config_factory=None) -> None:
        super().__init__(algo_name=algo_name)
        self.client_factory = client_factory or Client
        self.config_factory = config_factory or AxConfigurationFactory""")

# Update _configure_client to use self.config_factory
content = content.replace("ax_params = build_from_ax_parameters(ax_parameters)", "ax_params = self.config_factory.build_from_ax_parameters(ax_parameters)")
content = content.replace("ax_params = build_from_design_space(design_space, normalize)", "ax_params = self.config_factory.build_from_design_space(design_space, normalize)")
content = content.replace("ax_outcome_constraints = build_outcome_constraints(problem.constraints)", "ax_outcome_constraints = self.config_factory.build_outcome_constraints(problem.constraints)")
content = content.replace("opt_config = build_optimization_config(\n            ax_objectives, problem, ax_outcome_constraints\n        )", "opt_config = self.config_factory.build_optimization_config(\n            ax_objectives, problem, ax_outcome_constraints\n        )")

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
