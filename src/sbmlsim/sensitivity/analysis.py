"""Sensitivity analysis."""

import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import dill
import numpy as np
import pandas as pd
import roadrunner
import xarray as xr
from pymetadata.console import console
from rich.progress import track

from sbmlsim.sensitivity.parameters import SensitivityParameter
from sbmlsim.sensitivity.plots import heatmap


@dataclass
class SensitivityOutput:
    """Output measurement for SensitivityAnalysis."""

    uid: str
    name: str
    unit: Optional[str]


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

    def __init__(
        self,
        model_path: Path,
        selections: list[str],
        changes_simulation: dict[str, float],
        outputs: list[SensitivityOutput],
    ):
        self.model_path = model_path
        self.selections = selections
        self.changes_simulation = changes_simulation

        # store the simulation changes
        self.outputs: list[SensitivityOutput] = outputs

        # validate the outputs from the simulation
        rr = self.load_model(model_path=model_path, selections=self.selections)
        self.init_tolerances = list(rr.integrator.getAbsoluteToleranceVector())
        y = self.simulate(r=rr, changes={})
        outputs_dict = {q.uid for q in self.outputs}
        for key in y:
            if key not in outputs_dict:
                raise ValueError(
                    f"Key '{key}' missing in outputs dictionary: '{outputs_dict}"
                )

    @staticmethod
    def load_model(model_path: Path, selections: list[str]) -> roadrunner.RoadRunner:
        """Load roadrunner model."""
        rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        rr.selections = selections
        # integrator: roadrunner.Integrator = self.rr.integrator
        # integrator.setSetting("variable_step_size", True)
        return rr

    @staticmethod
    def apply_changes(
        r: roadrunner.RoadRunner, changes: dict[str, float], reset_all: bool = True
    ) -> None:
        """Apply changes after possible reset of the model."""
        if reset_all:
            r.resetAll()

        for key, value in changes.items():
            # print(f"{key=} {value=}")
            r.setValue(key, value)

    def simulate(
        self, r: roadrunner.RoadRunner, changes: dict[str, float]
    ) -> dict[str, float]:
        """Run a model simulation and return scalar results dictionary."""

        raise NotImplementedError

    @classmethod
    def parameter_values(
        cls,
        r: roadrunner.RoadRunner,
        parameters: list[SensitivityParameter],
        changes: dict[str, float],
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

        raise NotImplementedError


class SensitivityAnalysis:
    """Parent class for all sensitivity analysis."""

    def __init__(
        self,
        sensitivity_simulation: SensitivitySimulation,
        parameters: list[SensitivityParameter],
        groups: list[AnalysisGroup],
        results_path: Path,
        seed: Optional[int] = None,
        n_cores: Optional[int] = None,
        cache_results: bool = False,
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

        # storage and caching directory
        self.results_path: Path = results_path
        results_path.mkdir(parents=True, exist_ok=True)

        # set seed
        if seed is not None:
            np.random.seed(seed)

        # caching
        self.cache_results: bool = cache_results
        self.prefix: str = self.__class__.__name__

        # handle compute resources
        if not n_cores:
            n_cores = int(round(0.9 * multiprocessing.cpu_count()))
        self.n_cores = n_cores

        # parameter samples for sensitivity; shape: (num_samples x num_parameters)
        self.samples: dict[str, Optional[xr.DataArray]] = {}

        # outputs for given samples; shape: (num_samples x num_outputs)
        self.results: dict[str, Optional[xr.DataArray]] = {}

        # multiple sensitivities are stored
        # sensitivity matrix; shape: (num_parameters x num_outputs); could be multiple
        self.sensitivity: dict[str, dict[str, xr.DataArray]] = {
            g.uid: {} for g in self.groups
        }

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

    def execute(self):
        """Execute the sensitivity analysis."""
        console.rule(
            f"{self.__class__.__name__}",
            style="blue bold",
            align="center",
        )
        console.rule("Samples", style="white")
        self.create_samples()
        console.print(self.samples_table())

        console.rule("Results", style="white")
        self.simulate_samples(
            cache_filename=f"{self.prefix}_results.pkl",
            cache=self.cache_results,
        )
        console.print(self.results_table())

        console.rule("Sensitivity", style="white")
        self.calculate_sensitivity(
            cache_filename=f"{self.prefix}_sensitivity.pkl",
            cache=self.cache_results,
        )

    def create_samples(self) -> None:
        """Create and set parameter samples."""

        raise NotImplementedError

    @property
    def num_samples(self) -> int:
        """Number of samples.

        Requires that samples have been created.
        Assumes all groups have the same number of samples.
        """
        samples = self.samples[self.group_ids[0]]
        return samples.shape[0]

    def simulate_samples(
        self, cache_filename: Optional[str] = None, cache: bool = False
    ) -> None:
        """Simulate all samples in parallel.

        :param cache_filename: Path to the cache path.
        :param cache: If True, cache the simulated samples.
        """
        data = self.read_cache(cache_filename, cache)
        if data:
            self.results = data
            return

        for group in self.groups:
            console.print(f"Simulate group: '{group}'", style="blue")

            start = time.perf_counter()

            # num_samples x num_outputs
            results = xr.DataArray(
                np.full((self.num_samples, self.num_outputs), np.nan),
                dims=["sample", "output"],
                coords={"sample": range(self.num_samples), "output": self.outputs},
                name="results",
            )

            # load model
            r: roadrunner.RoadRunner = self.sensitivity_simulation.load_model(
                model_path=self.sensitivity_simulation.model_path,
                selections=self.sensitivity_simulation.selections,
            )

            # number of cores
            samples = self.samples[group.uid]

            # create chunk of samples for core
            def split_into_chunks(items, n):
                m = len(items)
                k, r = divmod(m, n)
                chunks = [
                    items[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)]
                    for i in range(n)
                ]
                chunked_samples = [
                    [
                        {
                            **group.changes,
                            **dict(zip(self.parameter_ids, samples[k, :].values)),
                        }
                        for k in chunk
                    ]
                    for chunk in chunks
                ]
                return chunks, chunked_samples

            items = list(range(self.num_samples))
            chunks, chunked_samples = split_into_chunks(items, self.n_cores)

            # parameters for multiprocessing
            sa_sim = self.sensitivity_simulation
            rrs = [(sa_sim, r, chunked_samples[i]) for i in range(self.n_cores)]

            with multiprocessing.Pool(processes=self.n_cores) as pool:
                outputs_list: list = pool.map(run_simulation, rrs)

            for kc, chunk in enumerate(chunks):
                for kp, idx in enumerate(chunk):
                    results[idx, :] = list(outputs_list[kc][kp].values())

            elapsed = time.perf_counter() - start
            self.results[group.uid] = results
            console.print(f"Parallel simulation: {elapsed:.3f} s")

        # write to cache
        self.write_cache(data=self.results, cache_filename=cache_filename, cache=cache)

    def calculate_sensitivity(
        self, cache_filename: Optional[str] = None, cache: bool = False
    ):
        """Calculate the sensitivity matrices."""

        raise NotImplementedError

    def samples_table(self) -> pd.DataFrame:
        return self._data_table(d=self.samples)

    def results_table(self) -> pd.DataFrame:
        return self._data_table(d=self.results)

    def _data_table(self, d: dict[str, xr.DataArray]) -> pd.DataFrame:
        items = []
        for group in self.groups:
            da: xr.DataArray = d[group.uid]
            item = {
                "group": group.uid,
                # 'group_name': group.name,
                **da.sizes,
            }
            items.append(item)
        return pd.DataFrame(items)

    def read_cache(self, cache_filename: str, cache: bool) -> Optional[Any]:
        cache_path: Optional[Path] = (
            self.results_path / cache_filename if cache_filename else None
        )
        if cache and not cache_path:
            raise ValueError("Cache path is required for caching.")

        # retrieve from cache
        if cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                data = dill.load(f)
                console.print(f"Simulated samples loaded from cache: '{cache_path}'")
                return data

        return None

    def write_cache(self, data: Any, cache_filename: str, cache: bool) -> Optional[Any]:
        cache_path: Optional[Path] = (
            self.results_path / cache_filename if cache_filename else None
        )
        if cache_path:
            with open(cache_path, "wb") as f:
                console.print(f"Simulated samples written to cache: '{cache_path}'")
                dill.dump(data, f)

    def sensitivity_df(self, group_id: str, key: str) -> pd.DataFrame:
        """Convert sensitivity information to dataframes."""

        sensitivity = self.sensitivity[group_id][key]
        return pd.DataFrame(
            sensitivity.values,
            columns=sensitivity.coords["output"],
            index=sensitivity.coords["parameter"],
        )

    def plot(self, **kwargs):
        """Should be implemented by subclass."""
        console.rule("Plotting", style="white")

    def plot_sensitivity(
        self,
        group_id: str,
        sensitivity_key: str,
        cutoff=0.1,
        cluster_rows: bool = True,
        title: Optional[str] = None,
        cmap: str = "seismic",
        fig_path: Optional[Path] = None,
        **kwargs,
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
            **kwargs,
        )


def run_simulation(params_tuple):
    """Pass all required arguments as parameter tuple."""
    sensitivity_simulation, r, chunked_changes = params_tuple
    outputs = []
    for kc in track(
        range(len(chunked_changes)), description=f"Simulate samples PID={os.getpid()}"
    ):
        changes = chunked_changes[kc]
        # console.print(f"PID={os.getpid()} | k={kc}")
        Y = sensitivity_simulation.simulate(r=r, changes=changes)
        outputs.append(Y)

    return outputs
