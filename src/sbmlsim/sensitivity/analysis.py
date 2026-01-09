"""Sensitivity analysis.

TODO implementation of alternative methods:
    - [ ] FAST
    - [ ] Morris
    - [ ] Sampling based methods (distribution)
"""
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from rich.progress import track
from pymetadata.console import console

import numpy as np
import pandas as pd
import xarray as xr

import roadrunner

import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol

from sbmlsim.sensitivity.parameters import SensitivityParameter
from sbmlsim.sensitivity.outputs import SensitivityOutput


class SensitivitySimulation:
    """Base class for sensitivity calculation.

    The sensitivity simulation runs a model simulation under a given set of
    model changes and returns a dictionary of scalar outputs.
    This function is called repeatedly during the sensitivity calculation.
    """

    def __init__(self, model_path: Path, selections: list[str], changes_simulation: dict[str, float], outputs: list[SensitivityOutput]):
        self.model_path = model_path
        self.selections = selections

        self.rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        self.rr.selections = self.selections
        # integrator: roadrunner.Integrator = self.rr.integrator
        # integrator.setSetting("variable_step_size", True)

        # store the simulation changes
        self.changes_simulation: dict[str, float] = changes_simulation
        self.outputs: list[SensitivityOutput] = outputs

        # validate the outputs from the simulation
        y = self.simulate(changes={})
        outputs_dict = {q.uid for q in self.outputs}
        for key in y:
            if key not in outputs_dict:
                raise ValueError(f"Key '{key}' missing in outputs dictionary: '{outputs_dict}")

    def apply_changes(self, changes: dict[str, float], reset_all: bool=True) -> None:
        """Apply changes after possible reset of the model."""
        if reset_all:
            self.rr.resetAll()
        for key, value in changes.items():
            # print(f"{key=} {value=}")
            self.rr.setValue(key, value)

    def simulate(self, changes: dict[str, float]) -> dict[str, float]:
        """Run a model simulation and return scalar results dictionary."""

        raise NotImplemented

    def parameter_values(self,
                         parameters: list[SensitivityParameter],
                         changes: dict[str, float]
                         ) -> dict[str, float]:
        """Get the parameter values for a given set of changes."""
        self.apply_changes(changes, reset_all=True)

        values: dict[str, float] = {}
        p: SensitivityParameter
        for p in parameters:
            values[p.uid] = self.rr.getValue(p.uid)

        return values

    def plot(self) -> None:
        """Plot the model simulation."""

        raise NotImplemented


class SensitivityAnalysis:
    """Parent class for all sensitivity analysis."""

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter]
                 ) -> None:
        """Create a sensitivity analysis for given parameter ids.

        Based on the results matrix the sensitivity is calculated.
        """
        self.sensitivity_simulation = sensitivity_simulation

        # parameters to vary; shape: (num_parameters,)
        self.parameters: list[SensitivityParameter] = parameters
        self.parameter_ids: list[str] = [p.uid for p in self.parameters]

        # outputs to calculate sensitivity on; shape: (num_outputs,)
        self.outputs: list[SensitivityOutput] = sensitivity_simulation.outputs
        self.output_ids: list[str] = [q.uid for q in self.outputs]

        # parameter samples for sensitivity; shape: (num_samples x num_parameters)
        self.samples: Optional[xr.DataArray] = None
        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: Optional[xr.DataArray] = None
        # sensitivity matrix; shape: (num_parameters x num_outputs); could be multiple
        self.sensitivity: Optional[xr.DataArray] = None
        self.sensitivity_normalized: Optional[xr.DataArray] = None


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

        # num_samples x num_outputs
        self.results = xr.DataArray(
            np.full((self.num_samples, self.num_outputs), np.nan),
            dims=["sample", "output"],
            coords={"sample": range(self.num_samples), "output": self.outputs},
            name="results"
        )

        pids = [p.uid for p in self.parameters]
        for k in track(range(self.num_samples), description="Simulating samples"):
            # console.print(f"{k}/{self.num_samples}")
            changes = dict(zip(pids, self.samples[k, :].values))
            # console.print(changes)
            outputs = self.sensitivity_simulation.simulate(changes=changes)
            self.results[k, :] = list(outputs.values())


    def calculate_sensitivity(self):
        """Calculate the sensitivity matrix."""

        raise NotImplemented



class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on local differences.

    param difference: change for calculation of local sensitivity (0.01 = 1% change)
    """

    def __init__(self, sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter], difference: float = 0.01):

        super().__init__(sensitivity_simulation, parameters)
        self.sensitivity: np.ndarray = np.zeros(shape=(self.num_parameters, self.num_outputs))
        self.difference: float = difference
        self.create_samples()

    @property
    def num_samples(self) -> int:
        """Number of parameter samples to simulate."""

        return 2 * self.num_parameters + 1

    def create_samples(self) -> None:
        """Create samples for the local sensitivity analysis.

        This requires a reference simulation and 2 simulations per parameter
        with increase and decrease of the respective parameter.
        """
        # Calculate the parameter values in the reference state
        parameter_values: dict[str, float] = self.sensitivity_simulation.parameter_values(
            parameters=self.parameters,
            changes=self.sensitivity_simulation.changes_simulation
        )

        # (num_samples x num_outputs)
        num_samples = 2 * self.num_parameters + 1
        samples = xr.DataArray(
            np.full((num_samples, self.num_parameters), np.nan),
            dims=["sample", "parameter"],
            coords={"sample": range(num_samples), "parameter": [p.uid for p in self.parameters]},
            name="samples"
        )

        reference_values = np.array(list(parameter_values.values()))
        for kp, pid in enumerate(parameter_values):
            value = parameter_values[pid]

            # right sided changes
            samples[2*kp, :] = reference_values
            samples[2*kp, kp] = value * (1.0 + self.difference)  # up
            samples[2 * kp + 1 , :] = reference_values
            samples[2 * kp + 1, kp] = value * (1.0 - self.difference) # down

        # reference values
        samples[-1, :] = reference_values # reference

        self.samples = samples

    def calculate_sensitivity(self):
        """Calculate the two-sided local sensitivity matrix."""

        # num_parameters x num_outputs
        # empty sensitivity
        self.sensitivity = xr.DataArray(
            np.full((self.num_parameters, self.num_outputs), np.nan),
            dims=["parameter", "output"],
            coords={"parameter": self.parameter_ids,
                    "output": self.output_ids},
            name="sensitivity"
        )
        self.sensitivity_normalized = xr.DataArray(
            np.full((self.num_parameters, self.num_outputs), np.nan),
            dims=["parameter", "output"],
            coords={"parameter": self.parameter_ids,
                    "output": self.output_ids},
            name="sensitivity"
        )

        for kp, p in enumerate(self.parameters):
            pid = self.parameters[kp].uid
            p_ref = self.samples[-1, kp]
            p_up = self.samples[2*kp, kp]
            p_down = self.samples[2 * kp + 1, kp]

            for ko, oid in enumerate(self.outputs):
                # num_samples x num_outputs
                q_ref = self.results[-1, ko]
                q_up = self.results[2*kp, ko]
                q_down = self.results[2 * kp + 1, ko]

                # two-sided sensitivity
                self.sensitivity[kp, ko] = (q_up - q_down) / (p_up - p_down)
                # normalized: relative change in output per relative change in parameter
                self.sensitivity_normalized[kp, ko] = self.sensitivity[kp, ko] * p_ref/q_ref

    @property
    def sensitivity_df(self) -> pd.DataFrame:
        """Convert sensitivity information to dataframe."""
        return pd.DataFrame(
            self.sensitivity_normalized.values,
            columns=self.sensitivity.coords["output"],
            index=self.sensitivity.coords["parameter"]
        )

    def plot_sensitivity(self, cutoff=0.1, cluster_rows: bool = True, title: Optional[str] = None):
        df = self.sensitivity_df
        self.plot_sensitivity_df(
            df=df,
            parameter_labels={p.uid: p.name for p in self.parameters},
            output_labels={q.uid: q.name for q in self.outputs},
            cutoff=cutoff,
            cluster_rows=cluster_rows,
            title=title
        )

    @staticmethod
    def plot_sensitivity_df(
        df: pd.DataFrame,
        parameter_labels: dict[str, str],
        output_labels: dict[str, str],
        cutoff=0.1, cluster_rows: bool = True,
        title: Optional[str] = None,
    ):
        from sbmlsim.sensitivity.plots import heatmap
        console.print(df)

        heatmap(
            df,
            parameter_labels=parameter_labels,
            output_labels=output_labels,
            cutoff=cutoff,
            cluster_rows=False,
            title=title,
        )


@dataclass
class SobolSensitivityAnalysis:
    """Global sensitivity analysis based on Sobol method.

    - [ ] SOBOL indices Sobol Sensitivity Analysis (Sobol 2001, Saltelli 2002, Saltelli et al. 2010)
      http://www.sciencedirect.com/science/article/pii/S0378475400002706
      https://www.sciencedirect.com/science/article/pii/S0010465502002801
      https://www.sciencedirect.com/science/article/pii/S0010465509003087
    """

    sensitivity_simulation: SensitivitySimulation

    def __init__(self, sensitivity_simulation: SensitivitySimulation):
        # assign simulation
        self.sensitivity_simulation = sensitivity_simulation

        self.sensitivity = np.zeros(shape=(self.num_parameters, self.num_outputs))

        # init the problem
        y = self.losartan_simulation(changes={})
        self.outputs = list(y.keys())
        self.names = ['BW']

        # Defining the model inputs
        sp = ProblemSpec({
            'num_vars': len(self.names),
            'names': self.names,
            'bounds': [
                [50, 150],
                # [0.003, 0.005]
            ],
            "outputs": self.outputs,
        })


        self.samples = self.create_samples()



    # def wrapped_run_simulation(self, X, func=losartan_simulation):
    #     # We transpose to obtain each column (the model factors) as separate variables
    #     changes: dict[str, float] = {}
    #     for k, key in enumerate(self.names):
    #         changes[key] = X[k]
    #
    #     # Then call the original model
    #     return list(func(self, changes).values())

    def create_samples(self):

        # libsa samples based on definition
        samples = saltelli.sample(sp, 1024)
        sp.set_samples(samples)

        # todo: transfer in standard simulation;


    def calculate_sensitivity(self):

        # transfer results in libsa results format
        Y = np.zeros((samples.shape[0], len(self.outputs)))
        for k, X in enumerate(samples):
            print(k)
            Y[k, :] = self.wrapped_run_simulation(X)
        sp.set_results(Y)

        # Perform Analysis
        Si = sp.analyze(SALib.analyze.sobol)

        # Store the sensitivity matrices
        print(Si['S1'])
        print(Si['ST'])
        total_Si, first_Si, second_Si = Si.to_df()


    def plot(self):
        Si.plot()
        from matplotlib import pyplot as plt
        plt.show()

