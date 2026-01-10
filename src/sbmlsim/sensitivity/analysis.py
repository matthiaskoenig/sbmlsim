"""Sensitivity analysis.

TODO implementation of alternative methods:
    - [ ] FAST
    - [ ] Morris
"""
import time
import multiprocessing
from typing import Optional, Any
from pathlib import Path
from rich.progress import track
from pymetadata.console import console

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import qmc

import roadrunner

import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol

from sbmlsim.sensitivity.parameters import SensitivityParameter
from sbmlsim.sensitivity.outputs import SensitivityOutput
from sbmlsim.sensitivity.plots import heatmap, sobol_barplot


class SensitivitySimulation:
    """Base class for sensitivity calculation.

    The sensitivity simulation runs a model simulation under a given set of
    model changes and returns a dictionary of scalar outputs.
    This function is called repeatedly during the sensitivity calculation.
    """

    def __init__(self, model_path: Path, selections: list[str],
                 changes_simulation: dict[str, float],
                 outputs: list[SensitivityOutput]):
        self.model_path = model_path
        self.selections = selections
        self.changes_simulation = changes_simulation

        # store the simulation changes
        self.outputs: list[SensitivityOutput] = outputs

        # validate the outputs from the simulation
        rr = self.load_model(model_path=model_path, selections=self.selections)
        y = self.simulate(r=rr, changes={})
        outputs_dict = {q.uid for q in self.outputs}
        for key in y:
            if key not in outputs_dict:
                raise ValueError(f"Key '{key}' missing in outputs dictionary: '{outputs_dict}")


    @staticmethod
    def load_model(model_path: Path, selections: list[str]) -> roadrunner.RoadRunner:
        """Load roadrunner model."""
        rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        rr.selections = selections
        # integrator: roadrunner.Integrator = self.rr.integrator
        # integrator.setSetting("variable_step_size", True)
        return rr

    @staticmethod
    def apply_changes(r: roadrunner.RoadRunner, changes: dict[str, float], reset_all: bool=True) -> None:
        """Apply changes after possible reset of the model."""
        if reset_all:
            r.resetAll()
        for key, value in changes.items():
            # print(f"{key=} {value=}")
            r.setValue(key, value)

    def simulate(self, r: roadrunner.RoadRunner, changes: dict[str, float]) -> dict[str, float]:
        """Run a model simulation and return scalar results dictionary."""

        raise NotImplemented

    @classmethod
    def parameter_values(cls, r: roadrunner.RoadRunner,
                         parameters: list[SensitivityParameter],
                         changes: dict[str, float]
                         ) -> dict[str, float]:
        """Get the parameter values for a given set of changes."""
        cls.apply_changes(r, changes, reset_all=True)

        values: dict[str, float] = {}
        p: SensitivityParameter
        for p in parameters:
            values[p.uid] = r.getValue(p.uid)

        return values

    def plot(self) -> None:
        """Plot the model simulation."""

        raise NotImplemented


class SensitivityAnalysis:
    """Parent class for all sensitivity analysis."""

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 results_path: Path,
                 ) -> None:
        """Create a sensitivity analysis for given parameter ids.

        Based on the results matrix the sensitivity is calculated.
        """
        self.sensitivity_simulation = sensitivity_simulation

        # outputs to calculate sensitivity on; shape: (num_outputs,)
        self.outputs: list[SensitivityOutput] = sensitivity_simulation.outputs
        self.output_ids: list[str] = [q.uid for q in self.outputs]

        # parameters to vary; shape: (num_parameters,)
        self.parameters: list[SensitivityParameter] = parameters
        self.parameter_ids: list[str] = [p.uid for p in self.parameters]

        # storage directory
        self.results_path: Path = results_path
        results_path.mkdir(parents=True, exist_ok=True)

        # parameter samples for sensitivity; shape: (num_samples x num_parameters)
        self.samples: Optional[xr.DataArray] = None
        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: Optional[xr.DataArray] = None

        # multiple sensitivities are stored
        # sensitivity matrix; shape: (num_parameters x num_outputs); could be multiple
        self.sensitivity: dict[str, xr.DataArray] = {}


    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def create_samples(self) -> None:
        """Create and set parameter samples."""

        raise NotImplemented

    @property
    def num_samples(self) -> int:
        """Number of samples.

        Requires that samples have been created.
        """
        return self.samples.shape[0]

    def simulate_samples(self) -> None:
        """Simulate all samples."""
        start = time.perf_counter()

        # num_samples x num_outputs
        self.results = xr.DataArray(
            np.full((self.num_samples, self.num_outputs), np.nan),
            dims=["sample", "output"],
            coords={"sample": range(self.num_samples), "output": self.outputs},
            name="results"
        )

        # load the integrators
        r: roadrunner.RoadRunner = self.sensitivity_simulation.load_model(
            model_path=self.sensitivity_simulation.model_path,
            selections=self.sensitivity_simulation.selections,
        )

        for k in track(range(self.num_samples), description="Simulating samples"):
            changes = dict(zip(self.parameter_ids, self.samples[k, :].values))
            outputs = self.sensitivity_simulation.simulate(
                r=r,
                changes=changes
            )
            self.results[k, :] = list(outputs.values())

        elapsed = time.perf_counter() - start
        console.print(f"Serial: {elapsed:.3f} s")

    def simulate_samples_parallel(self) -> None:
        """Simulate all samples in parallel."""
        start = time.perf_counter()

        # num_samples x num_outputs
        self.results = xr.DataArray(
            np.full((self.num_samples, self.num_outputs), np.nan),
            dims=["sample", "output"],
            coords={"sample": range(self.num_samples), "output": self.outputs},
            name="results"
        )

        # load model
        r: roadrunner.RoadRunner = self.sensitivity_simulation.load_model(
            model_path=self.sensitivity_simulation.model_path,
            selections=self.sensitivity_simulation.selections,
        )

        # number of cores
        n_cores = multiprocessing.cpu_count()

        # create chunk of samples for core
        def split_into_chunks(items, n):
            m = len(items)
            k, r = divmod(m, n)
            chunks = [
                items[i * k + min(i, r):(i + 1) * k + min(i + 1, r)]
                for i in range(n)
            ]
            chunked_samples = [
                [dict(zip(self.parameter_ids, self.samples[k, :].values)) for k in chunk]
                for chunk in chunks
            ]
            return chunks, chunked_samples

        items = list(range(self.num_samples))
        chunks, chunked_samples = split_into_chunks(items, n_cores)

        # parameters for multiprocessing
        sa_sim = self.sensitivity_simulation
        rrs = [(sa_sim, r, chunked_samples[i]) for i in range(n_cores)]

        with multiprocessing.Pool(processes=n_cores) as pool:
            outputs_list: list = pool.map(run_simulation, rrs)

        for kc, chunk in enumerate(chunks):
            for kp, idx in enumerate(chunk):
                self.results[idx, :] = list(outputs_list[kc][kp].values())

        elapsed = time.perf_counter() - start
        console.print(f"Parallel simulation: {elapsed:.3f} s")


    def calculate_sensitivity(self):
        """Calculate the sensitivity matrices."""

        raise NotImplemented

    def sensitivity_df(self, key="normalized") -> pd.DataFrame:
        """Convert sensitivity information to dataframe."""

        return pd.DataFrame(
            self.sensitivity[key].values,
            columns=self.sensitivity[key].coords["output"],
            index=self.sensitivity[key].coords["parameter"]
        )

    def plot_sensitivity(
        self,
        key: str, cutoff=0.1,
        cluster_rows: bool = True,
        title: Optional[str] = None,
        cmap: str = "seismic",
        fig_path: Optional[Path] = None,
        **kwargs
    ) -> None:
        df = self.sensitivity_df(key=key)
        heatmap(
            df=df,
            parameter_labels={p.uid: f"{p.uid}: {p.name}" for p in self.parameters},
            output_labels={q.uid: q.name for q in self.outputs},
            cutoff=cutoff,
            cluster_rows=cluster_rows,
            title=title,
            cmap=cmap,
            fig_path=fig_path,
            **kwargs
        )

import os

def run_simulation(
    params_tuple
):
    """Pass all required arguments as parameter tuple."""
    sensitivity_simulation, r, chunked_changes = params_tuple
    outputs = []
    for kc in track(range(len(chunked_changes)), description=f"Simulate samples PID={os.getpid()}"):
        changes = chunked_changes[kc]
        # console.print(f"PID={os.getpid()} | k={kc}")
        Y = sensitivity_simulation.simulate(
            r=r,
            changes=changes
        )
        outputs.append(Y)

    return outputs


class LocalSensitivityAnalysis(SensitivityAnalysis):
    """Local sensitivity analysis based on local differences.

    param difference: change for calculation of local sensitivity (0.01 = 1% change)
    """

    def __init__(self, sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 results_path: Path,
                 difference: float = 0.01):

        super().__init__(sensitivity_simulation, parameters, results_path)

        self.difference: float = difference

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
        r = self.sensitivity_simulation.load_model(self.sensitivity_simulation.model_path, selections=self.sensitivity_simulation.selections)
        parameter_values: dict[str, float] = self.sensitivity_simulation.parameter_values(
            r=r,
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
        console.print(self.samples)

    def calculate_sensitivity(self):
        """Calculate the two-sided local sensitivity matrix."""

        # num_parameters x num_outputs
        for key in ["raw", "normalized"]:
            self.sensitivity[key] = xr.DataArray(
            np.full((self.num_parameters, self.num_outputs), np.nan),
            dims=["parameter", "output"],
            coords={"parameter": self.parameter_ids,
                    "output": self.output_ids},
            name=key
        )

        sensitivity_raw = self.sensitivity["raw"]
        sensitivity_normalized = self.sensitivity["normalized"]

        for kp, p in enumerate(self.parameters):
            p_ref = self.samples[-1, kp]
            p_up = self.samples[2*kp, kp]
            p_down = self.samples[2 * kp + 1, kp]

            for ko, oid in enumerate(self.outputs):
                # num_samples x num_outputs
                q_ref = self.results[-1, ko]
                q_up = self.results[2*kp, ko]
                q_down = self.results[2 * kp + 1, ko]

                # two-sided sensitivity
                sensitivity_raw[kp, ko] = (q_up - q_down) / (p_up - p_down)
                # normalized: relative change in output per relative change in parameter
                sensitivity_normalized[kp, ko] = sensitivity_raw[kp, ko] * p_ref/q_ref


class SobolSensitivityAnalysis(SensitivityAnalysis):
    """Global sensitivity analysis based on Sobol method.

    - [ ] SOBOL indices Sobol Sensitivity Analysis (Sobol 2001, Saltelli 2002, Saltelli et al. 2010)
      http://www.sciencedirect.com/science/article/pii/S0378475400002706
      https://www.sciencedirect.com/science/article/pii/S0010465502002801
      https://www.sciencedirect.com/science/article/pii/S0010465509003087
    """

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 N: int,
                 results_path: Path,
                 ):

        super().__init__(sensitivity_simulation, parameters, results_path)
        self.N: int = N

        # define the problem specification
        self.ssa_problem: ProblemSpec = ProblemSpec({
            'num_vars': self.num_parameters,
            'names': self.parameter_ids,
            'bounds': [ [p.lower_bound, p.upper_bound] for p in self.parameters],
            "outputs": self.output_ids,
        })
        # console.print(self.ssa_problem)


    def create_samples(self) -> None:
        """Create samples for sobol.

        Generates model inputs using Saltelli's extension of the Sobol' sequence

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """

        # libsa samples based on definition
        ssa_samples = saltelli.sample(self.ssa_problem, N=self.N, calc_second_order=True)
        self.ssa_problem.set_samples(ssa_samples)

        # (num_samples x num_outputs)
        #  total model evaluations are (2d+2) * N for d input factors
        num_samples = (2 * self.num_parameters + 2) * self.N

        self.samples = xr.DataArray(
            ssa_samples,
            dims=["sample", "parameter"],
            coords={"sample": range(num_samples),
                    "parameter": self.parameter_ids},
            name="samples"
        )


    def calculate_sensitivity(self) -> None:
        """Calculate the sensitivity matrices."""

        Y = self.results.values
        self.ssa_problem.set_results(Y)

        # num_parameters x num_outputs
        sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]
        for key in sensitivity_keys:
            self.sensitivity[key] = xr.DataArray(
                np.full((self.num_parameters, self.num_outputs), np.nan),
                dims=["parameter", "output"],
                coords={"parameter": self.parameter_ids,
                        "output": self.output_ids},
                name=key
            )

        # Perform Analysis
        # Si is a Python dict-like with the keys "S1", "S2", "ST",
        # "S1_conf", "S2_conf", and "ST_conf".
        # The _conf keys store the corresponding confidence intervals,
        # typically with a confidence level of 95%.

        # Calculate Sobol indices for every output
        for ko in range(self.num_outputs):
            Yo = Y[:, ko]
            Si = SALib.analyze.sobol.analyze(
                self.ssa_problem, Yo,
                calc_second_order=True,
                print_to_console=False,
                n_processors=4,
            )
            for key in sensitivity_keys:
                self.sensitivity[key][:, ko] = Si[key]


    def plot_sobol_indices(
        self,
        fig_path: Path,
        **kwargs
        ):
        """Barplots for the Sobol indices.

        """
        # parameter_labels: dict[str, str] = {p.uid: f"{p.uid}: {p.name}" for p in self.parameters}
        parameter_labels: dict[str, str] = {p.uid: p.uid for p in self.parameters}
        output_labels: dict[str, str] = {q.uid: q.name for q in self.outputs}

        ymax = self.sensitivity["ST"].max(dim=None)
        ymin = self.sensitivity["S1"].min(dim=None)
        console.print(f"{ymax=}")

        for ko, output in enumerate(self.outputs):
            f_path = fig_path.parent / f"{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"

            S1 = self.sensitivity["S1"][:, ko]
            ST = self.sensitivity["ST"][:, ko]
            S1_conf = self.sensitivity["S1_conf"][:, ko]
            ST_conf = self.sensitivity["ST_conf"][:, ko]
            console.print(S1)
            console.print(type(S1))
            sobol_barplot(
                S1=S1,
                ST=ST,
                S1_conf=S1_conf,
                ST_conf=ST_conf,
                title=output_labels[output.uid],
                fig_path=f_path,
                parameter_labels=parameter_labels,
                ymax=np.max([1.05, ymax]),
                ymin=np.min([-0.05, ymin]),
            )

class SamplingSensitivityAnalysis(SensitivityAnalysis):
    """Sensitivity/uncertainty analysis based on sampling."""

    sensitivity_keys = [
        "mean",
        "median",
        "std",
        "cv",
        "min",
        "q005",
        "q095",
        "max"
    ]

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 N: int,
                 results_path: Path,
                 ):

        super().__init__(sensitivity_simulation, parameters, results_path)
        self.N: int = N


    def create_samples(self) -> None:
        """Create LHS samples.

        Latin hypercube sampling (LHS) is a stratified sampling method used to
        generate near‑random samples from a multidimensional distribution for Monte
        Carlo simulations and computer experiments.

        Use LHS sampling of parameters.
        """
        # LHS sampling (uniform distributed in bounds)
        sampler = qmc.LatinHypercube(d=self.num_parameters)  # number of dimensions
        u = sampler.random(n=self.N)  # shape (n, d), in [0, 1], number of samples

        # Scale to parameter bounds
        lower = np.array([p.lower_bound for p in self.parameters])
        upper = np.array([p.upper_bound for p in self.parameters])
        x = qmc.scale(u, lower, upper)

        self.samples = xr.DataArray(
            x,
            dims=["sample", "parameter"],
            coords={"sample": range(self.N),
                    "parameter": self.parameter_ids},
            name="samples"
        )

    def calculate_sensitivity(self) -> None:
        """Calculate the sensitivity matrices."""

        # calculate readouts
        for key in self.sensitivity_keys:
            self.sensitivity[key] = xr.DataArray(
                np.full(self.num_outputs, np.nan),
                dims=["output"],
                coords={
                    "output": self.output_ids},
                name=key
            )

        for ko, oid in enumerate(self.outputs):
            # num_samples x num_outputs
            data = self.results.values[:, ko]
            for key in self.sensitivity_keys:
                if key == "mean":
                    value = np.mean(data)
                elif key == "median":
                    value = np.median(data)
                elif key == "std":
                    value = np.std(data)
                elif key == "cv":
                    value = np.std(data)/np.mean(data)
                elif key == "min":
                    value = np.min(data)
                elif key == "q005":
                    value = np.quantile(data, q=0.05)
                elif key == "q095":
                    value = np.quantile(data, q=0.95)
                elif key == "max":
                    value = np.max(data)
                else:
                    raise KeyError(key)

                self.sensitivity[key][ko] = value

    def df_sampling_sensitivity(
        self,
        df_path: Path,
    ):
        # dataframe with the values
        items = []
        for ko, output in enumerate(self.outputs):
            item: dict[str, Any] = {
                "uid": output.uid,
                "name": output.name,
                "N": self.N,
            }
            for key in self.sensitivity_keys:
                item[key] = self.sensitivity[key].values[ko]
            item["unit"] = output.unit

            items.append(item)

        df = pd.DataFrame(items)
        console.print(df)
        if df_path:
            df.to_csv(df_path, index=False, sep="\t")

            # latex table
            latex_path = df_path.parent / f"{df_path.stem}.tex"
            df_latex: pd.DataFrame = df.copy()
            df_latex.drop('uid', axis=1, inplace=True)
            df_latex.to_latex(latex_path, index=False, float_format="{:.3g}".format)

        return df



    def plot_sampling_sensitivity(
        self,
        fig_path: Path,
        **kwargs
        ):
        """Boxplots for the Sampling sensitivity."""

        # width
        figsize = (15, 15)
        label_fontsize = 15
        from matplotlib import pyplot as plt
        ncols = np.ceil(np.sqrt(self.num_outputs))
        n_empty = ncols*ncols - self.num_outputs
        n_empty_rows = np.floor(n_empty/ncols)

        nrows = ncols-n_empty_rows


        f, axes = plt.subplots(figsize=figsize, nrows=int(nrows), ncols=int(ncols), layout="constrained")
        for ko, ax in enumerate(axes.flat):
            if ko > self.num_outputs-1:
                ax.axis('off')
            else:

                output = self.outputs[ko]
                data = self.results.values[:, ko]

                # outliers for scatter
                # Q1 = np.percentile(data, 25)
                # Q3 = np.percentile(data, 75)
                # IQR = Q3 - Q1
                # lower_fence = Q1 - 1.5 * IQR
                # upper_fence = Q3 + 1.5 * IQR
                # data_no_outliers = data[(data > lower_fence) & (data < upper_fence)]
                data_no_outliers = data

                ax.boxplot(data, positions=[0.2],  # labels=[output.name],
                           patch_artist=True, showfliers=False,
                           boxprops=dict(
                               facecolor='lightblue',
                               alpha=0.7
                           )
                )
                # ax.violinplot(data, positions=[0.8], showmeans=True,
                #                showmedians=True,
                #                showextrema = False
                #               )
                # jitter_width = 0.05  # Adjust for spacing
                # x_jitter = np.random.normal(0.8, jitter_width, len(data_no_outliers))
                # ax.scatter(x_jitter, data_no_outliers, alpha=0.7, s=30, color='darkgrey',
                #                edgecolors='black'
                # )

                # ax.set_xlabel('Parameter', fontsize=label_fontsize, fontweight="bold")
                # ax.set_ylim(bottom=0)
                # ax.set_title(output.name, fontsize=15, fontweight="bold")
                ax.set_ylabel(f"{output.name} [{output.unit}]", fontsize=label_fontsize, fontweight="bold")
                ax.tick_params(axis='x', which='both', labelbottom=False)
                # ax.grid(True, axis="y")
                # ax.tick_params(axis='x', labelrotation=90)

        # if title:
        #     plt.suptitle(title, fontsize=20, fontweight="bold")
        if fig_path:
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
