"""Sensitivity analysis.

TODO implementation of alternative methods:
    - [ ] FAST
    - [ ] Morris
"""
import time
import multiprocessing
from dataclasses import dataclass
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


@dataclass
class AnalysisGroup:
    """Subgroup for analysis."""

    uid: str
    name: str
    changes: dict[str, float]
    color: Optional[str]


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
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 seed: Optional[int]=None,
                 ) -> None:
        """Create a sensitivity analysis for given parameter ids.

        Based on the results matrix the sensitivity is calculated.
        """
        self.sensitivity_simulation = sensitivity_simulation

        # outputs to calculate sensitivity on; shape: (num_outputs,)
        self.outputs: list[SensitivityOutput] = sensitivity_simulation.outputs

        # parameters to vary; shape: (num_parameters,)
        self.parameters: list[SensitivityParameter] = parameters

        # groups for analysis
        self.groups: list[AnalysisGroup] = groups

        # remove parameters which are set in the base simulation or group
        # sensitivity does not make sense on these
        fixed_parameters = set()
        for pid in sensitivity_simulation.changes_simulation.keys():
            fixed_parameters.add(pid)
        for group in self.groups:
            for pid in group.changes.keys():
                fixed_parameters.add(pid)
        for p in self.parameters:
            if p.uid in fixed_parameters:
                console.print(f"Removing fixed parameter: {p.uid}", style="warning")
        self.parameters = [p for p in self.parameters if p.uid not in fixed_parameters]

        # storage directory
        self.results_path: Path = results_path
        results_path.mkdir(parents=True, exist_ok=True)

        # set seed
        if seed is not None:
            np.random.seed(seed)

        # parameter samples for sensitivity; shape: (num_samples x num_parameters)
        self.samples: dict[str, Optional[xr.DataArray]] = {}

        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: dict[str, Optional[xr.DataArray]] = {}

        # multiple sensitivities are stored
        # sensitivity matrix; shape: (num_parameters x num_outputs); could be multiple
        self.sensitivity: dict[str, dict[str, xr.DataArray]] = {g.uid: {} for g in self.groups}

    @property
    def output_ids(self) -> list[str]:
        return [o.uid for o in self.outputs]

    @property
    def parameter_ids(self) -> list[str]:
        return [p.uid for p in self.parameters]

    @property
    def group_ids(self) -> list[str]:
        return [g.uid for g in self.groups]

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def create_samples(self) -> None:
        """Create and set parameter samples."""

        raise NotImplemented

    @property
    def num_samples(self) -> int:
        """Number of samples.

        Requires that samples have been created.
        Assumes all groups have the same number of samples.
        """
        samples = self.samples[self.group_ids[0]]
        return samples.shape[0]

    def simulate_samples(self) -> None:
        """Simulate all samples in parallel."""

        for group in self.groups:
            console.print(f"Simulate group: '{group}'", style="blue")

            start = time.perf_counter()

            # num_samples x num_outputs
            results = xr.DataArray(
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

            samples = self.samples[group.uid]

            # create chunk of samples for core
            def split_into_chunks(items, n):
                m = len(items)
                k, r = divmod(m, n)
                chunks = [
                    items[i * k + min(i, r):(i + 1) * k + min(i + 1, r)]
                    for i in range(n)
                ]
                chunked_samples = [
                    [{
                        **group.changes,
                        **dict(zip(self.parameter_ids, samples[k, :].values))
                    } for k in chunk]
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
                    results[idx, :] = list(outputs_list[kc][kp].values())

            elapsed = time.perf_counter() - start
            self.results[group.uid] = results
            console.print(f"Parallel simulation: {elapsed:.3f} s")


    def calculate_sensitivity(self):
        """Calculate the sensitivity matrices."""

        raise NotImplemented

    def sensitivity_df(self, group_id: str, key: str) -> pd.DataFrame:
        """Convert sensitivity information to dataframes."""

        sensitivity = self.sensitivity[group_id][key]
        return pd.DataFrame(
            sensitivity.values,
            columns=sensitivity.coords["output"],
            index=sensitivity.coords["parameter"]
        )

    def plot_sensitivity(
        self,
        group_id: str,
        sensitivity_key: str,
        cutoff=0.1,
        cluster_rows: bool = True,
        title: Optional[str] = None,
        cmap: str = "seismic",
        fig_path: Optional[Path] = None,
        **kwargs
    ) -> None:

        df = self.sensitivity_df(group_id=group_id, key=sensitivity_key)
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
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 difference: float = 0.01,
                 **kwargs) -> None:

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)

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
        for group in self.groups:

            # Calculate the parameter values in the reference state
            r = self.sensitivity_simulation.load_model(self.sensitivity_simulation.model_path, selections=self.sensitivity_simulation.selections)

            # parameter values require simulation and group changes to be applied
            parameter_values: dict[str, float] = self.sensitivity_simulation.parameter_values(
                r=r,
                parameters=self.parameters,
                changes={
                    **self.sensitivity_simulation.changes_simulation,
                    **group.changes,
                }
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

            self.samples[group.uid] = samples

        console.print(self.samples)

    def calculate_sensitivity(self):
        """Calculate the two-sided local sensitivity matrix."""

        for gid in self.group_ids:
            # num_parameters x num_outputs
            for key in ["raw", "normalized"]:
                self.sensitivity[gid][key] = xr.DataArray(
                np.full((self.num_parameters, self.num_outputs), np.nan),
                dims=["parameter", "output"],
                coords={"parameter": self.parameter_ids,
                        "output": self.output_ids},
                name=key
            )

            sensitivity_raw = self.sensitivity[gid]["raw"]
            sensitivity_normalized = self.sensitivity[gid]["normalized"]

            samples = self.samples[gid]
            results = self.results[gid]

            for kp, p in enumerate(self.parameters):
                p_ref = samples[-1, kp]
                p_up = samples[2*kp, kp]
                p_down = samples[2 * kp + 1, kp]

                for ko, oid in enumerate(self.outputs):
                    # num_samples x num_outputs
                    q_ref = results[-1, ko]
                    q_up = results[2*kp, ko]
                    q_down = results[2 * kp + 1, ko]

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

    sensitivity_keys = ["S1", "ST", "S1_conf", "ST_conf"]

    def __init__(self,
                 sensitivity_simulation: SensitivitySimulation,
                 parameters: list[SensitivityParameter],
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 N: int,
                 **kwargs,
                 ):

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)
        self.N: int = N

        # define the problem specification
        self.ssa_problems: dict[str, ProblemSpec] = {}
        for group in self.groups:
            self.ssa_problems[group.uid] = ProblemSpec({
                'num_vars': self.num_parameters,
                'names': self.parameter_ids,
                'bounds': [ [p.lower_bound, p.upper_bound] for p in self.parameters],
                "outputs": self.output_ids,
            })

    def create_samples(self) -> None:
        """Create samples for sobol.

        Generates model inputs using Saltelli's extension of the Sobol' sequence

        The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
        to generate uniform samples of parameter space.
        """

        # (num_samples x num_outputs)
        #  total model evaluations are (2d+2) * N for d input factors
        num_samples = (2 * self.num_parameters + 2) * self.N

        for gid in self.group_ids:
            # libsa samples based on definition
            ssa_samples = saltelli.sample(self.ssa_problems[gid], N=self.N, calc_second_order=True)
            self.ssa_problems[gid].set_samples(ssa_samples)

            self.samples[gid] = xr.DataArray(
                ssa_samples,
                dims=["sample", "parameter"],
                coords={"sample": range(num_samples),
                        "parameter": self.parameter_ids},
                name="samples"
            )


    def calculate_sensitivity(self) -> None:
        """Calculate the sensitivity matrices."""

        for gid in self.group_ids:
            Y = self.results[gid].values
            self.ssa_problems[gid].set_results(Y)

            # num_parameters x num_outputs

            for key in self.sensitivity_keys:
                self.sensitivity[gid][key] = xr.DataArray(
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
                    self.ssa_problems[gid], Yo,
                    calc_second_order=True,
                    print_to_console=False,
                    n_processors=4,
                )
                for key in self.sensitivity_keys:
                    self.sensitivity[gid][key][:, ko] = Si[key]


    def plot_sobol_indices(
        self,
        fig_path: Path,
        ):
        """Barplots for the Sobol indices."""
        # parameter_labels: dict[str, str] = {p.uid: f"{p.uid}: {p.name}" for p in self.parameters}
        parameter_labels: dict[str, str] = {p.uid: p.uid for p in self.parameters}
        output_labels: dict[str, str] = {q.uid: q.name for q in self.outputs}

        for group in self.groups:
            gid = group.uid
            ymax = self.sensitivity[gid]["ST"].max(dim=None)
            ymin = self.sensitivity[gid]["S1"].min(dim=None)

            for ko, output in enumerate(self.outputs):
                # f_path = fig_path.parent / f"FigS{ko+22}_{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"
                f_path = fig_path.parent / f"{fig_path.stem}_{ko:>03}_{output.uid}{fig_path.suffix}"

                S1 = self.sensitivity[gid]["S1"][:, ko]
                ST = self.sensitivity[gid]["ST"][:, ko]
                S1_conf = self.sensitivity[gid]["S1_conf"][:, ko]
                ST_conf = self.sensitivity[gid]["ST_conf"][:, ko]
                sobol_barplot(
                    S1=S1,
                    ST=ST,
                    S1_conf=S1_conf,
                    ST_conf=ST_conf,
                    title=f"{output_labels[output.uid]} ({group.name})",
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
                 groups: list[AnalysisGroup],
                 results_path: Path,
                 N: int,
                 **kwargs,
                 ):

        super().__init__(sensitivity_simulation, parameters, groups, results_path, **kwargs)
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
        lower = np.array([p.lower_bound for p in self.parameters])
        upper = np.array([p.upper_bound for p in self.parameters])

        for gid in self.group_ids:
            u = sampler.random(n=self.N)  # shape (n, d), in [0, 1], number of samples
            self.samples[gid] = xr.DataArray(
                qmc.scale(u, lower, upper),  # scale to parameter bounds
                dims=["sample", "parameter"],
                coords={"sample": range(self.N),
                        "parameter": self.parameter_ids},
                name="samples"
            )

    def calculate_sensitivity(self) -> None:
        """Calculate the sensitivity matrices."""
        for gid in self.group_ids:

            # calculate readouts
            for key in self.sensitivity_keys:
                self.sensitivity[gid][key] = xr.DataArray(
                    np.full(self.num_outputs, np.nan),
                    dims=["output"],
                    coords={
                        "output": self.output_ids},
                    name=key
                )

            for ko, oid in enumerate(self.outputs):
                # num_samples x num_outputs
                data = self.results[gid].values[:, ko]
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

                    self.sensitivity[gid][key][ko] = value

    def df_sampling_sensitivity(
        self,
        df_path: Path,
    ):
        # dataframe with the values
        items = []
        for group in self.groups:
            for ko, output in enumerate(self.outputs):
                item: dict[str, Any] = {
                    "gid": group.uid,
                    "gname": group.name,
                    "uid": output.uid,
                    "name": output.name,
                    "N": self.N,
                }
                for key in self.sensitivity_keys:
                    item[key] = self.sensitivity[group.uid][key].values[ko]
                item["unit"] = output.unit

                items.append(item)

        df = pd.DataFrame(items)
        console.print(df)
        if df_path:
            df.to_csv(df_path, index=False, sep="\t")

            # latex table
            latex_path = df_path.parent / f"{df_path.stem}.tex"
            df_latex: pd.DataFrame = df.copy()
            df_latex.drop(['gid', 'uid', 'N', "min", "max", "q005", "q095"], axis=1, inplace=True)
            latex_str = df_latex.to_latex(None, index=False, float_format="{:.3g}".format)
            latex_str = latex_str.replace("∞", r"$\infty$")
            latex_str = latex_str.replace("*", r"$\cdot$")

            with open(latex_path, "w") as f:
                f.write(latex_str)

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
