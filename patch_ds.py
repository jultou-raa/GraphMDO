import re

with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

ds_build_code = """    @staticmethod
    def build_from_design_space(
        design_space: Any, normalize: bool
    ) -> list[RangeParameterConfig]:
        ax_params = []
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            l_b, u_b, _, _ = get_value_and_bounds(
                design_space, var_name, normalize=normalize
            )
            # Introspect GEMSEO variable types to preserve integers
            var_type_list = design_space.variable_types.get(var_name, [])
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                # Determine if this specific component is an integer
                is_int = False
                if i < len(var_type_list) and var_type_list[i] == type(1):
                    is_int = True

                # If normalizing, we usually force float because normalized bounds are 0.0 to 1.0
                p_type = "int" if (is_int and not normalize) else "float"

                if p_type == "int":
                    ax_params.append(
                        RangeParameterConfig(
                            name=param_name, bounds=(int(l_b[i]), int(u_b[i])), parameter_type=p_type
                        )
                    )
                else:
                    ax_params.append(
                        RangeParameterConfig(
                            name=param_name, bounds=(float(l_b[i]), float(u_b[i])), parameter_type=p_type
                        )
                    )
        return ax_params"""

start_idx = content.find("@staticmethod\n    def build_from_design_space(")
end_idx = content.find("@staticmethod\n    def build_optimization_config(", start_idx)

content = content[:start_idx] + ds_build_code + "\n\n    " + content[end_idx:]

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(content)
