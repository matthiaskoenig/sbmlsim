"""
Subclass of RoadRunner with additional functionality.

Provides common functionality used in multiple simulation scenarios, like
- making selections
- timed simulations
- setting integrator settings
- setting values in model
- plotting
All simulations should be run via the MyRoadRunner class which provides
additional tests on the method.

Set of unittests provides tested functionality.
"""

# TODO: checkout the OPTIMIZE Settings
# roadrunner.Config.setValue(roadrunner.Config.OPTIMIZE_REACTION_RATE_SELECTION, True)
# named matrix
# roadrunner.Config.setValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX, False)

# TODO: proper subsets of selections
# TODO: test the difference of init between boundary and floating species.

from __future__ import print_function, division
import warnings
import libsbml
import numpy
from roadrunner import RoadRunner, SelectionRecord
from pandas import DataFrame
from multiscale.util.timing import time_it


class MyRunner(RoadRunner):
    """
    Subclass of RoadRunner with additional functionality.
    """
    # TODO: the same init has to happen when a model is reloaded/loaded via
    # different means like load function
    @time_it(message="SBML compile")
    def __init__(self, *args, **kwargs):
        """
        See RoadRunner() information for arguments.
        :param args:
        :param kwargs:
        :return:
        """
        # super constructor
        RoadRunner.__init__(self, *args, **kwargs)
        self.debug = False
        # set the default settings for the integrator
        self.set_default_settings()
        # provide access to SBMLDocument
        self.sbml_doc = libsbml.readSBMLFromString(self.getCurrentSBML())

    #########################################################################
    # Settings
    #########################################################################
    def set_debug(self, debug):
        self.debug = debug

    def set_default_settings(self):
        """ Set default settings of integrator. """
        self.set_integrator_settings(
                variable_step_size=True,
                stiff=True,
                absolute_tolerance=1E-8,
                relative_tolerance=1E-8
        )

    def set_integrator_settings(self, **kwargs):
        """ Set integrator settings. """
        for key, value in kwargs.items():
            # adapt the absolute_tolerance relative to the amounts
            if key == "absolute_tolerance":
                value = value * min(self.model.getCompartmentVolumes())
            self.integrator.setValue(key, value)

        if self.debug:
            print(self.integrator)

    #########################################################################
    # Simulation
    #########################################################################
    # supported simulate options
    _simulate_args = frozenset(["start", "end", "steps", "selections"])

    @time_it()
    def simulate(self, *args, **kwargs):
        """ Timed simulate function. """
        for key in kwargs:
            assert(key in self._simulate_args)
        return RoadRunner.simulate(self, *args, **kwargs)

    def simulate_complex(self, concentrations={}, amounts={}, parameters={},
                         reset_parameters=False, **kwargs):
        """ Perform RoadRunner simulation.
            Sets parameter values given in parameters dictionary &
            initial values provided in dictionaries.

            :param amounts:
                dictionary of initial_amounts (overwrites initial concentrations)
            :param concentrations:
                dictionary of initial_concentrations
            :param parameters:
                dictionary of parameter changes
            :param reset_parameters:
                reset the parameter changes after the simulation

            :returns tuple of simulation result and global parameters at end point of
                    simulation (<NamedArray>, <DataFrame>)
        """
        # TODO: fixme (clear setting of concentrations).
        # change parameters & recalculate initial assignments
        if len(parameters) > 0:
            old_concentrations = self.store_concentrations()
            old_parameters = self._set_parameters(parameters)
            self.reset(SelectionRecord.INITIAL_GLOBAL_PARAMETER)
            self._set_concentrations(old_concentrations)

        # set changed concentrations
        if len(concentrations) > 0:
            self._set_initial_concentrations(concentrations)
        # set changed amounts
        if len(amounts) > 0:
            self._set_initial_amounts(amounts)

        # simulate
        if time_it:
            s = self.simulate(**kwargs)
        else:
            s = RoadRunner.simulate(self, **kwargs)

        # reset parameters
        if reset_parameters:
            self._set_parameters(old_parameters)
            self.reset(SelectionRecord.INITIAL_GLOBAL_PARAMETER)

        # return simulation time course
        return s

    #########################################################################
    # Setting & storing model values
    #########################################################################
    @classmethod
    def check_keys(cls, keys, key_type):
        import re
        if key_type == "INITIAL_CONCENTRATION":
            pattern = "^init\(\[\w+\]\)$"
        elif key_type == "INITIAL_AMOUNT":
            pattern = "^init\(\w+\)$"
        elif key_type == "CONCENTRATION":
            pattern = "^\[\w+\]$"
        elif key_type in ["AMOUNT", "PARAMETER"]:
            pattern = "^(?!init)\w+$"
        else:
            raise KeyError("Key type not supported.")

        for key in keys:
            assert(re.match(pattern, key))

    def store_concentrations(self):
        """
        Store FloatingSpecies concentrations of current model state.
        :return: {sid: ci} dictionary of concentrations
        """
        return {"[{}]".format(sid): self["[{}]".format(sid)] for sid in self.model.getFloatingSpeciesIds()}

    def store_amounts(self):
        """
        Store FloatingSpecies amounts of current model state.
        :return: {sid: ci} dictionary of amounts
        """
        return {sid: self[sid] for sid in self.model.getFloatingSpeciesIds()}

    def _set_values(self, value_dict):
        """
        Set values in model from {selection: value}.
        :return: {selection: original} returns dictionary of original values.
        """
        changed = dict()
        for key, value in value_dict.items():
            changed[key] = self[key]
            self[key] = value
        return changed

    def _set_parameters(self, parameters):
        """
        Set parameters in model from {sid: value}.
        :return: {sid: original} returns dictionary of original values.
        """
        self.check_keys(parameters.keys(), "PARAMETER")
        return self._set_values(parameters)

    def _set_initial_concentrations(self, concentrations):
        """
        Set initial concentration in model from {init([sid]): value}.
        :return: {init([sid]): original} returns dictionary of original values.
        """
        self.check_keys(concentrations.keys(), "INITIAL_CONCENTRATION")
        return self._set_values(concentrations)

    def _set_concentrations(self, concentrations):
        """
        Set concentrations in model from {[sid]: value}.
        :return: {[sid]: original} returns dictionary of original values.
        """
        self.check_keys(concentrations.keys(), "CONCENTRATION")
        return self._set_values(concentrations)

    def _set_initial_amounts(self, amounts):
        """
        Set initial amounts in model from {init(sid): value}.
        :return: {init(sid): original} returns dictionary of original values.
        """
        self.check_keys(amounts.keys(), "INITIAL_AMOUNT")
        return self._set_values(amounts)

    def _set_amounts(self, amounts):
        """
        Set amounts in model from {sid: value}.
        :return: {sid: original} returns dictionary of original values.
        """
        self.check_keys(amounts.keys(), "AMOUNT")
        return self._set_values(amounts)

    #########################################################################
    # Helper for units & selections
    #########################################################################
    # TODO: create some frozenset for fast checking
    # self.parameters = frozenset(self.)

    def selections_floating_concentrations(self):
        """
        Set floating concentration selections in RoadRunner.
            list[str] of selections for time, [c1], ..[cN]
        """
        self.selections = ['time'] + sorted(['[{}]'.format(s) for s in self.model.getFloatingSpeciesIds()])

    def selections_floating_amounts(self):
        """
        Set floating amount selections in RoadRunner.
            list[str] of selections for time, c1, ..cN
        """
        self.selections = ['time'] + sorted(self.model.getFloatingSpeciesIds())

    #########################################################################
    # DataFrames
    #########################################################################
    # TODO: add rules and assignments

    def df_global_parameters(self):
        """
        Create GlobalParameter DataFrame.
        :return: pandas DataFrame
        """
        sids = self.model.getGlobalParameterIds()
        model = self.sbml_doc.getModel()
        parameters = [model.getParameter(sid) for sid in sids]
        df = DataFrame({
            'value': self.model.getGlobalParameterValues(),
            'unit': [p.units for p in parameters],
            'constant': [p.constant for p in parameters],
            'parameter': parameters,
            'name': [p.name for p in parameters],
            }, index=sids, columns=['value', 'unit', 'constant', 'parameter', 'name'])
        return df

    def df_species(self):
        """
        Create FloatingSpecies DataFrame.
        :return: pandas DataFrame
        """
        sids = self.model.getFloatingSpeciesIds() + self.model.getBoundarySpeciesIds()
        model = self.sbml_doc.getModel()
        species = [model.getSpecies(sid) for sid in sids]
        df = DataFrame({
            'concentration': numpy.concatenate([self.model.getFloatingSpeciesConcentrations(),
                                                self.model.getBoundarySpeciesConcentrations()],
                                               axis=0),
            'amount': numpy.concatenate([self.model.getFloatingSpeciesAmounts(),
                                         self.model.getBoundarySpeciesAmounts()],
                                         axis=0),
            'unit': [s.units for s in species],
            'constant': [s.constant for s in species],
            'boundaryCondition': [s.boundary_condition for s in species],
            'species': species,
            'name': [s.name for s in species],
            }, index=sids, columns=['concentration', 'amount', 'unit', 'constant', 'boundaryCondition', 'species', 'name'])
        return df

    def df_simulation(self):
        """
        DataFrame of the simulation data.
        :return: pandas DataFrame
        """
        df = DataFrame(self.getSimulationData(),
                       columns=self.selections)
        return df

    #########################################################################
    # Plotting
    #########################################################################
    @classmethod
    def plot_results(cls, results, *args, **kwargs):
        """
        :param results: list of result matrices
        :return:
        """
        import matplotlib.pylab as plt

        plt.figure(figsize=(7, 4))
        for s in results:
            plt.plot(s[:, 0], s[:, 1:], *args, **kwargs)
            # print('tend:', s[-1, 0])
        # labels
        plt_fontsize = 30
        plt.xlabel('time [s]', fontsize=plt_fontsize)
        plt.ylabel('species [mM]', fontsize=plt_fontsize)
