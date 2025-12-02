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

    def __init__(self, model_path: Path, selections: list[str]):
        self.model_path = model_path
        self.selections = selections
        self.rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        self.rr.selections = self.selections
        integrator: roadrunner.Integrator = self.rr.integrator
        integrator.setSetting("variable_step_size", True)
        # state = rr.saveStateS()

        # get the outputs from the simulation
        y = self.simulate(changes={})
        self.outputs = list(y.keys())


    def simulate(self, changes: dict[str, float]) -> dict[str, float]:
        """Runs a model simulation and returns the scalar results dictionary.

        This must be implemented by the subclass to work.
        """
        raise NotImplemented

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
class SBMLSensitivityAnalysis:
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
