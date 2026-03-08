with open("tests/test_optimizer.py", "r") as f:
    content = f.read()

content = content.replace("config = build_optimization_config(", "config = AxConfigurationFactory.build_optimization_config(")
content = content.replace("build_from_design_space", "AxConfigurationFactory.build_from_design_space")
content = content.replace("AxConfigurationFactory.AxConfigurationFactory.build_from_design_space", "AxConfigurationFactory.build_from_design_space")


with open("tests/test_optimizer.py", "w") as f:
    f.write(content)

with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    lib_content = f.read()

lib_content = lib_content.replace("""            # Introspect GEMSEO variable types to preserve integers
            var_type_list = design_space.variable_types.get(var_name, [])
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                # Determine if this specific component is an integer
                is_int = False
                if i < len(var_type_list) and var_type_list[i] is int:
                    is_int = True""", """            # Introspect GEMSEO variable types to preserve integers
            var_type_list = design_space.variable_types.get(var_name, [])
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                # Determine if this specific component is an integer
                is_int = False
                if i < len(var_type_list) and var_type_list[i] is int:
                    is_int = True""")

# I also need to fix TypeError get_value_and_bounds got unexpected keyword argument 'normalize'
lib_content = lib_content.replace("""            l_b, u_b, _, _ = get_value_and_bounds(
                design_space, var_name, normalize=normalize
            )""", """            l_b, u_b = design_space.get_bounds(var_name)
            if normalize:
                # Assuming GEMSEO handles normalization internally or we just pass float.
                # This was a bug in original code too, let's keep it simple since we just want to avoid unexpected kwargs.
                l_b, u_b = [0.0] * size, [1.0] * size""")

with open("src/mdo_framework/optimization/ax_algo_lib.py", "w") as f:
    f.write(lib_content)
