with open("src/mdo_framework/optimization/ax_algo_lib.py", "r") as f:
    content = f.read()

# Since we use module-level functions now, we shouldn't have `AxParameterBuilder.` etc.
# Wait! In `full_rewrite.py`, I did the replacement BEFORE the unindenting?
# I DID `content = content.replace(p_builder, new_p_builder)` which means `AxParameterBuilder` is gone.
# But where is `build_from_ax_parameters` called?
# Inside `_configure_client`. Let's check `ax_algo_lib.py` and see where they are defined.
