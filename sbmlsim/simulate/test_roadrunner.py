"""
Testing the expected roadrunner behaviour.
"""
import unittest
import roadrunner
import tellurium as te
import roadrunner_tools as rt
from multiscale.examples.testdata import demo_sbml


class RoadrunnerTestCase(unittest.TestCase):

    def test_simulation_fixed_steps2(self):
        """ Test fixed step size simulation. """
        import roadrunner
        r1 = roadrunner.RoadRunner(demo_sbml)
        s1 = r1.simulate(0, 10)
        self.assertFalse(r1.getIntegrator().getSetting('variable_step_size'))
        self.assertEqual(51, s1.shape[0])
        self.assertEqual(7, s1.shape[1])

        r2 = rt.MyRunner(demo_sbml)
        r2.selections_floating_concentrations()
        r2.set_integrator_settings(variable_step_size=False)
        s2 = r2.simulate(0, 10)
        self.assertFalse(r1.getIntegrator().getSetting('variable_step_size'))
        self.assertEqual(51, s2.shape[0])
        self.assertEqual(7, s2.shape[1])

    def test_simulation_variable_steps(self):
        """ Test variable step size simulation. """
        r = rt.MyRunner(demo_sbml)
        r.integrator.setSetting('variable_step_size', True)
        r.selections_floating_concentrations()
        s = r.simulate_complex(start=0, end=20)

        self.assertTrue(r.getIntegrator().getSetting('variable_step_size'))
        self.assertNotEqual(101, s.shape[0])
        self.assertEqual(7, s.shape[1])
        self.assertEqual(s['time'][0], 0.0)
        self.assertEqual(s['time'][-1], 20.0)

    def test_simulate_parameters(self):
        """ Test setting parameters in model. """
        r = rt.MyRunner(demo_sbml)
        r.selections = ['time', 'Vmax_bA', 'Vmax_bB']
        parameters = {'Vmax_bA': 10.0, 'Vmax_bB': 7.15}
        s = r.simulate_complex(start=0, end=20, parameters=parameters)
        df_gp = r.df_global_parameters()
        self.assertEqual(10.0, df_gp.value['Vmax_bA'])
        self.assertEqual(7.15, df_gp.value['Vmax_bB'])
        self.assertEqual(10.0, s['Vmax_bA'][0])
        self.assertEqual(7.15, s['Vmax_bB'][0])

    def test_simulate_initial_concentrations(self):
        """ Test setting initial concentrations in model. """
        r = rt.MyRunner(demo_sbml)
        concentrations = {'init([e__A])': 5.0, 'init([e__B])': 2.0}
        s = r.simulate_complex(start=0, end=20, concentrations=concentrations)
        self.assertEqual(5.0, s['[e__A]'][0])
        self.assertEqual(2.0, s['[e__B]'][0])

    def test_simulate_initial_amounts(self):
        """ Test setting initial amounts in model. """
        r = rt.MyRunner(demo_sbml)
        r.selections = ['time', 'e__A', 'e__B']
        amounts = {'init(e__A)': 0.01, 'init(e__B)': 0.004}
        s = r.simulate_complex(start=0, end=20, amounts=amounts)
        self.assertEqual(0.01, s['e__A'][0])
        self.assertEqual(0.004, s['e__B'][0])

    def test_selections(self):
        """ Test the standard selection of roadrunner. """
        r = rt.MyRunner(demo_sbml)
        self.assertEqual(len(r.selections), 7)
        self.assertTrue('time' in r.selections)
        self.assertTrue('[e__A]' in r.selections)
        self.assertTrue('[e__B]' in r.selections)
        self.assertTrue('[e__C]' in r.selections)
        self.assertTrue('[c__A]' in r.selections)
        self.assertTrue('[c__B]' in r.selections)
        self.assertTrue('[c__C]' in r.selections)

if __name__ == '__main__':
    unittest.main()
