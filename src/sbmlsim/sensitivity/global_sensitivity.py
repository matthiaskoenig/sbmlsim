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
import xarray as xr

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
    parameters: list[str]

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
        self.samples: Optional[xr.DataArray] = None
        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: Optional[xr.DataArray] = None
        # sensitivity matrix; shape: (num_parameters x num_outputs); could be multiple
        self.sensitivity_results: Optional[xr.DataArray] = None

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

        for k in range(self.num_samples):
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

        super().__init__(sensitivity_simulation, parameters)
        self.sensitivity = np.zeros(shape=(self.num_parameters, self.num_outputs))
        self.difference = difference
        self.samples = self.create_samples()

        # TODO: flag left-sided, right-sided, both-sided

    @property
    def num_samples(self) -> int:
        """Number of parameter samples to simulate."""
        return 2 * self.num_parameters

    def create_samples(self) -> None:

        # Calculate the parameter values in the reference state
        parameter_values: dict[str, float] = self.sensitivity_simulation.parameter_values(
            parameters=self.parameters,
            changes=self.sensitivity_simulation.changes_simulation
        )

        # (num_samples x num_outputs)
        num_samples = 2*self.num_parameters
        samples = np.empty(shape=(num_samples, self.num_parameters))
        samples = xr.DataArray(
            np.full((num_samples, self.num_parameters), np.nan),
            dims=["sample", "parameter"],
            coords={"sample": range(num_samples), "parameter": self.parameters},
            name="samples"
        )

        reference_values = np.array(list(parameter_values.values()))
        for kp, pid in enumerate(parameter_values):
            value = parameter_values[pid]

            # right sided changes
            samples[2*kp, :] = reference_values
            samples[2*kp, kp] = value * (1.0 + self.difference)
            samples[2 * kp + 1 , :] = reference_values
            samples[2 * kp + 1, :] = value * (1.0 - self.difference)

        self.samples = samples

    def calculate_sensitivity(self):

        pass

    def plot_sensitivity(self):

        pass

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def heatmap(da: xr.DataArray, cutoff: float=0.01, annotate_values=True, transpose: bool=False):
    """Creates heatmap of model sensitivity"""

    def calculate_mask(df, cutoff=0.01):
        """Calculates a boolean mask DataFrame for the heatmap based on cutoff."""
        mask = np.empty(shape=df.shape, dtype="bool")
        for index, value in np.ndenumerate(df):
            if np.abs(value) < cutoff:
                mask[index] = True
            else:
                mask[index] = False
        return pd.DataFrame(data=mask, columns=df.COLUMNS, index=df.index)

    def calculate_subset(df, cutoff=0.01):
        """Calculates subset of data frame consisting of rows where at least
        one value is above cutoff."""
        return df[(df.abs() >= cutoff).any(axis=1)]



    # filter rows
    # X.drop(pk_exclude, axis=1, inplace=True)

    # if cutoff > 0:
    # X_subset = calculate_subset(X, cutoff=cutoff)
    # X_subset_mask = calculate_mask(X_subset, cutoff)
    da_subset = da

    # yticklabels = ["{}".format(pid) for pid in X_subset.index]
    # xticklabels = ["{}".format(pnames[pid]["label"]) for pid in X_subset.COLUMNS]

    xticklabels = da.coords[da.dims[1]]
    yticklabels = da.coords[da.dims[0]]

    # plot heatmap
    ax = sns.clustermap(
        da_subset,
        center=0,
        # vmin=-0.2,
        # vmax=0.2,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="seismic",
        cbar_pos=(0.05, 0.25, 0.03, 0.4),
        annot=annotate_values,
        fmt="1.2f",
        annot_kws={"size": 13},
        # mask=X_subset_mask,
        col_cluster=False,
        method="single",
        figsize=(20, 20),
    )
    plt.setp(
        ax.ax_heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        size=20,
    )
    plt.setp(ax.ax_heatmap.get_yticklabels(), size=20)
    ax.ax_cbar.tick_params(labelsize=20)
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_col_dendrogram.set_visible(False)

    # create custom legend containing yticklabels and their description
    # handles = [t.get_text() for t in ax.ax_heatmap.get_yticklabels()]
    # labels = [pnames[pid]["label"] for pid in handles]
    #
    # # FIXME: update after defining labels
    # idx = [pnames[pid]["idx"] for pid in handles]
    # # idx = [k for k, pid in enumerate(handles)]
    #
    # labels = [label for _, label in sorted(zip(idx, labels))]
    # handles = [f"{handle}:" for _, handle in sorted(zip(idx, handles))]
    # handles = [handle.replace("_", "\_") for handle in handles]

    # mid = int(np.ceil(len(handles) / 2))
    # legend1 = plt.legend(
    #     handles[:mid],
    #     labels[:mid],
    #     handler_map={str: LegendTitle({"fontsize": 16})},
    #     fontsize=16,
    #     frameon=False,
    #     bbox_to_anchor=(1.2, -0.6),
    #     loc="upper left",
    #     handlelength=14,
    # )
    # legend2 = plt.legend(
    #     handles[mid:],
    #     labels[mid:],
    #     handler_map={str: LegendTitle({"fontsize": 16})},
    #     fontsize=16,
    #     frameon=False,
    #     bbox_to_anchor=(13, -0.6),
    #     loc="upper left",
    #     handlelength=19,
    # )
    # plt.gca().add_artist(legend1)

    # plt.savefig(
    #     results_dir / "parameter.sensitivity_cluster.png", dpi=300, bbox_inches="tight"
    # )
    # plt.savefig(results_dir / "parameter.sensitivity_cluster.svg", bbox_inches="tight")

    plt.show()




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


    def calculate_sensitivity(self):

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

        # Generate samples
        samples = saltelli.sample(sp, 1024)
        sp.set_samples(samples)


        # Evaluate model
        # sp.evaluate(wrapped_run_simulation)

        Y = np.zeros((samples.shape[0], len(self.outputs)))
        for k, X in enumerate(samples):
             print(k)
             Y[k, :] = self.wrapped_run_simulation(X)
        sp.set_results(Y)


        # Perform Analysis
        Si = sp.analyze(SALib.analyze.sobol)
        print(Si['S1'])
        print(Si['ST'])
        total_Si, first_Si, second_Si = Si.to_df()
        Si.plot()
        from matplotlib import pyplot as plt
        plt.show()




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
