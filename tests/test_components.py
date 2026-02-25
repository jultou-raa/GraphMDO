import unittest
import openmdao.api as om
from mdo_framework.core.components import ToolComponent

def simple_func(x, y):
    return x + y

class TestToolComponent(unittest.TestCase):
    def test_simple_execution(self):
        prob = om.Problem()

        comp = ToolComponent(
            name='add',
            func=simple_func,
            inputs=['x', 'y'],
            outputs=['z']
        )

        prob.model.add_subsystem('comp', comp, promotes=['*'])

        prob.setup()

        prob.set_val('x', 2.0)
        prob.set_val('y', 3.0)

        prob.run_model()

        self.assertAlmostEqual(prob.get_val('z')[0], 5.0)

    def test_dict_output(self):
        def dict_func(a):
            return {'b': a * 2, 'c': a + 1}

        prob = om.Problem()

        comp = ToolComponent(
            name='dict_comp',
            func=dict_func,
            inputs=['a'],
            outputs=['b', 'c']
        )

        prob.model.add_subsystem('comp', comp, promotes=['*'])

        prob.setup()

        prob.set_val('a', 10.0)

        prob.run_model()

        self.assertAlmostEqual(prob.get_val('b')[0], 20.0)
        self.assertAlmostEqual(prob.get_val('c')[0], 11.0)

    def test_tuple_output(self):
        def tuple_func(a):
            return a * 2, a + 1

        prob = om.Problem()

        comp = ToolComponent(
            name='tuple_comp',
            func=tuple_func,
            inputs=['a'],
            outputs=['b', 'c']
        )

        prob.model.add_subsystem('comp', comp, promotes=['*'])

        prob.setup()

        prob.set_val('a', 10.0)

        prob.run_model()

        self.assertAlmostEqual(prob.get_val('b')[0], 20.0)
        self.assertAlmostEqual(prob.get_val('c')[0], 11.0)

    def test_derivatives_setup(self):
        # Just verifying that declare_partials is called with exact method
        prob = om.Problem()

        comp = ToolComponent(
            name='deriv_comp',
            func=simple_func,
            inputs=['x', 'y'],
            outputs=['z'],
            derivatives=True
        )

        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()
        # No easy way to check internal state without mocking openmdao internals,
        # but execution should pass.
        prob.run_model()

    def test_compute_partials(self):
        # Call compute_partials manually to cover the lines
        comp = ToolComponent(
            name='deriv_comp',
            func=simple_func,
            inputs=['x', 'y'],
            outputs=['z'],
            derivatives=True
        )

        # We need to initialize options manually if not running via setup
        # But easier to let setup handle it.
        prob = om.Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()

        # Manually call compute_partials
        inputs = {'x': 1.0, 'y': 2.0}
        partials = {} # Dummy
        comp.compute_partials(inputs, partials)
        # Should just pass (pass statement in code)

    def test_positional_arguments_fallback(self):
        # Test function that doesn't accept kwargs, to trigger fallback logic
        def pos_only(x, y):
            # This function signature accepts positional args.
            # But calling it with func(**{'x': 1, 'y': 2}) works fine in Python.
            # We need a case where kwargs fail.
            # Python functions usually accept kwargs matching argument names.
            # Unless we force positional-only arguments (Python 3.8+).
            return x + y

        # However, to test the try-except TypeError block in components.py:
        # try: result = func(**input_vals) except TypeError: result = func(*input_vals.values())
        # We need func(**input_vals) to RAISE TypeError.

        # This happens if input_vals keys don't match function arguments.
        # BUT ToolComponent logic maps inputs dict keys from self.options['inputs'].
        # So if we configure component with inputs=['x', 'y'], input_vals has keys 'x', 'y'.

        # If the function is defined as def foo(a, b), calling foo(x=1, y=2) raises TypeError.
        # This is the case we want to test: Component inputs names mismatch function argument names,
        # but function relies on positional order.

        def mismatch_args(a, b):
            return a + b

        prob = om.Problem()

        comp = ToolComponent(
            name='mismatch',
            func=mismatch_args,
            inputs=['x', 'y'], # Mismatch 'a', 'b'
            outputs=['z']
        )

        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()

        prob.set_val('x', 1.0)
        prob.set_val('y', 2.0)

        # This should trigger the fallback to positional args
        # Because mismatch_args(x=1, y=2) raises TypeError.
        # Then mismatch_args(1, 2) succeeds.
        prob.run_model()

        self.assertAlmostEqual(prob.get_val('z')[0], 3.0)

if __name__ == '__main__':
    unittest.main()
