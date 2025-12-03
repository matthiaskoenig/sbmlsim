"""Global sensitivity analysis.

TODO:

- [ ] get defined parameter bounds from model (annotate information);
- [ ] storage of results simulation
- [ ] storage of results sensitivity analysis
- [ ] visualization of results (heatmap)
- [ ] alternative methods:
    - [ ] local sensitivity analysis
    - [ ] SOBOL indices Sobol Sensitivity Analysis (Sobol 2001, Saltelli 2002, Saltelli et al. 2010)
          http://www.sciencedirect.com/science/article/pii/S0378475400002706
          https://www.sciencedirect.com/science/article/pii/S0010465502002801
          https://www.sciencedirect.com/science/article/pii/S0010465509003087
    - [ ] FAST
    - [ ] Morris
    - [ ] Sampling based methods (distribution)
- [ ] report as PDF with references and description (Typst)
- [ ] parallelization ? (benchmark)

"""
from typing import Optional

import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

import numpy as np
import roadrunner
from pathlib import Path
from dataclasses import dataclass


from sbmlutils.console import console
from roadrunner._roadrunner import NamedArray


@dataclass
class SensitivitySimulation:
    """Base class for sensitivity calculation.

    The sensitivity simulation runs a model simulation under a given set of
    model changes and returns a dictionary of scalar outputs.
    This function is called repeatedly during the sensitivity calculation.
    """

    model_path: Path
    selections: list[str]
    rr: roadrunner.RoadRunner = None
    outputs: list[str] = None
    changes_simulation: dict[str, float] = None

    def __init__(self, model_path: Path, selections: list[str], changes_simulation: dict[str, float]):
        self.model_path = model_path
        self.selections = selections
        self.rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        self.rr.selections = self.selections
        integrator: roadrunner.Integrator = self.rr.integrator
        integrator.setSetting("variable_step_size", True)
        # state = rr.saveStateS()

        # store the simulation changes
        self.changes_simulation = changes_simulation

        # get the outputs from the simulation
        y = self.simulate(changes={})
        self.outputs = list(y.keys())


    def simulate(self, changes: dict[str, float]) -> dict[str, float]:
        """Runs a model simulation and returns the scalar results dictionary.

        This must be implemented by the subclass to work.
        """
        raise NotImplemented

    def parameter_values(self, parameters: list[str], changes: dict[str, float]) -> dict[str, float]:
        """Get the parameter values for a given set of changes."""
        self.apply_changes(changes, reset_all=True)
        values: dict[str, float] = {}
        for pid in parameters:
            values[pid] = self.rr.getValue(pid)
        return values


    def plot(self) -> None:
        """Plots the model simulation for debugging."""
        raise NotImplemented

    def apply_changes(self, changes: dict[str, float], reset_all: bool=True) -> None:
        """Apply changes after possible reset of the model."""
        if reset_all:
            self.rr.resetAll()
        for key, value in changes.items():
            # print(f"{key=} {value=}")
            self.rr.setValue(key, value)


@dataclass
class SensitivityAnalysis:
    """Parent class for all sensitivity analysis.

    TODO: additional metadata for the outputs and the parameters; i.e. name, units, bounds, ....
    """

    sensitivity_simulation: SensitivitySimulation

    def __init__(self, sensitivity_simulation: SensitivitySimulation,
                 parameters: list[str]) -> None:
        """Create a sensitivity analysis for given parameter ids.

        Based on the results matrix the sensitivity is calculated.
        """
        self.sensitivity_simulation = sensitivity_simulation

        # parameters to vary; shape: (num_parameters,)
        self.parameters: list[str] = parameters
        # outputs to calculate sensitivity on; shape: (num_outputs,)
        self.outputs: list[str] = sensitivity_simulation.outputs
        # parameter samples for sensitivity; shape: (num_samples x num_parameters)
        self.samples: Optional[np.ndarray] = None
        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: Optional[np.ndarray] = None
        # sensitivity matrix; shape: (num_parameters x num_outputs)
        self.sensitivity_results: Optional[np.ndarray] = None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def create_samples(self) -> None:
        """Create and set parameter samples."""

        raise NotImplemented

    def num_samples(self) -> int:
        """Number of samples.

        Requires that samples have been created.
        """
        return self.samples.shape[0]

    def simulate_samples(self) -> None:
        """Simulate all samples."""
        self.samples = np.zeros(shape=(self.num_samples, self.num_parameters))
        self.outputs = np.zeros(shape=(self.num_samples, self.num_outputs))

        for k in range(self.num_samples()):
            changes = dict(zip(self.parameters, self.samples[k, :]))
            outputs = self.sensitivity_simulation.simulate(changes=changes)
            self.outputs[k, :] = outputs

    def calculate_sensitivity(self):
        """Calculate the sensitivity matrix."""

        raise NotImplemented

@dataclass
class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on local differences."""

    difference: float
    sensitivity: np.ndarray = None

    def __init__(self, sensitivity_simulation: SensitivitySimulation,
                 parameters: list[str], difference: float = 0.1):

        self.sensitivity = np.zeros(shape=(self.num_parameters, self.num_outputs))
        self.difference = difference
        self.samples = self.create_samples()

    @property
    def num_samples(self) -> int:
        """Number of parameter samples to simulate."""
        return 2 * self.num_parameters

    def create_samples(self) -> np.ndarray:

        for key, value in p_ref.items():
            values = np.ones(shape=(2 * num_pars,)) * value.magnitude
            # change parameters in correct position
            values[index] = value.magnitude * (1.0 + difference)
            values[index + num_pars] = value.magnitude * (1.0 - difference)
            changes[key] = Q_(values, value.units)
            index += 1

    def calculate_sensitivity(self):

        pass

    def plot_sensitivity(self):

        pass


@dataclass
class SamplingSensitivityAnalysis(SensitivityAnalysis):
    """Sample from provided parameter distributions."""

    # TODO: implement
    pass

@dataclass
class GlobalSobolSensitivityAnalysis:
    """Parent class for sensitivity analysis."""

    sensitivity_simulation: SensitivitySimulation

    def __init__(self, sensitivity_simulation: SensitivitySimulation):
        # assign simulation
        self.sensitivity_simulation = sensitivity_simulation


    # def wrapped_run_simulation(self, X, func=losartan_simulation):
    #     # We transpose to obtain each column (the model factors) as separate variables
    #     changes: dict[str, float] = {}
    #     for k, key in enumerate(self.names):
    #         changes[key] = X[k]
    #
    #     # Then call the original model
    #     return list(func(self, changes).values())


    # def calculate_sensitivity(self):
    #
    #     y = self.losartan_simulation(changes={})
    #     self.outputs = list(y.keys())
    #     self.names = ['BW']
    #
    #     # Defining the model inputs
    #     sp = ProblemSpec({
    #         'num_vars': len(self.names),
    #         'names': self.names,
    #         'bounds': [
    #             [50, 150],
    #             # [0.003, 0.005]
    #         ],
    #         "outputs": self.outputs,
    #     })
    #
    #     # Generate samples
    #     samples = saltelli.sample(sp, 1024)
    #     sp.set_samples(samples)
    #
    #
    #     # Evaluate model
    #     # sp.evaluate(wrapped_run_simulation)
    #
    #     Y = np.zeros((samples.shape[0], len(self.outputs)))
    #     for k, X in enumerate(samples):
    #          print(k)
    #          Y[k, :] = self.wrapped_run_simulation(X)
    #     sp.set_results(Y)
    #
    #
    #     # Perform Analysis
    #     Si = sp.analyze(SALib.analyze.sobol)
    #     print(Si['S1'])
    #     print(Si['ST'])
    #     total_Si, first_Si, second_Si = Si.to_df()
    #     Si.plot()
    #     from matplotlib import pyplot as plt
    #     plt.show()




if __name__ == '__main__':
    model_path = Path(__file__).parent / "models" / "losartan" / "losartan_body_flat.xml"

    sa = SBMLSensitivityAnalysis(
        model_path=model_path,
        selections=[
            "time",
            "[Cve_los]",
            "[Cve_e3174]",
            "[Cve_l158]",
            "[ang1]",
            "[ang2]",
            "[ren]",
            "[ald]",
            "SBP",
            "DBP",
            "MAP",
        ]
    )
    y = sa.losartan_simulation(changes={})
    console.print(y)

    # y = run_simulation()
    # print(y)
    sa.calculate_sensitivity()
