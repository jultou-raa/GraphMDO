import openmdao.api as om


class ToolComponent(om.ExplicitComponent):
    """
    Generic OpenMDAO component that wraps a Python function.
    """

    def initialize(self):
        self.options.declare("name", types=str, desc="Name of the tool")
        self.options.declare("func", types=object, desc="The function to execute")
        self.options.declare("inputs", types=list, desc="List of input variable names")
        self.options.declare(
            "outputs", types=list, desc="List of output variable names"
        )
        self.options.declare(
            "derivatives",
            types=bool,
            default=False,
            desc="Whether analytic derivatives are provided",
        )

    def setup(self):
        inputs = self.options["inputs"]
        outputs = self.options["outputs"]

        for input_name in inputs:
            self.add_input(input_name, val=0.0)

        for output_name in outputs:
            self.add_output(output_name, val=0.0)

        # Declare partials
        if self.options["derivatives"]:
            self.declare_partials("*", "*", method="exact")
        else:
            self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        func = self.options["func"]
        # Prepare inputs as a dictionary
        input_vals = {name: inputs[name] for name in self.options["inputs"]}

        # Execute the function
        # We assume the function returns a dictionary or a single value
        try:
            result = func(**input_vals)
        except TypeError:
            # Fallback if function expects positional arguments (simple wrappers)
            result = func(*input_vals.values())

        # Map results to outputs
        if len(self.options["outputs"]) == 1:
            output_name = self.options["outputs"][0]
            outputs[output_name] = result
        else:
            if isinstance(result, dict):
                for name in self.options["outputs"]:
                    outputs[name] = result[name]
            else:
                # If result is a tuple/list, assume order matches outputs
                for i, name in enumerate(self.options["outputs"]):
                    outputs[name] = result[i]

    def compute_partials(self, inputs, partials):
        if self.options["derivatives"]:
            # If the function provides derivatives, we would call it here.
            # This is a placeholder as the signature for derivative function varies.
            pass
