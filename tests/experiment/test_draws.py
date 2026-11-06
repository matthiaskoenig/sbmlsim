"""The simulation experiments over the draws of the fitted parameters of a project.

The draws are those of the toy project's model whose rate is the value of the
adopted fit of the toy project with its uncertainty (`tests.project_toy.records`).
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from sbmlsim import Q, run
from sbmlsim.data import Data
from sbmlsim.distrib import UNCERTAINTY_DIMENSION
from sbmlsim.experiment import ExperimentRunError, ExperimentRunner
from sbmlsim.experiment.draws import (
    UncertaintyRun,
    UncertaintySettings,
    _over_draws,
    check_fit_parameters,
)
from sbmlsim.fit.records import AdoptedRecords
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Curve, Figure
from sbmlsim.plot.plotting import Band
from sbmlsim.project import FIGURES_DIR, Project, write_adopted
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import point_timecourse
from sbmlsim.run.kinds import ConfigurationError
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulation.sampling import uncertainty_dimension
from sbmlsim.simulator import Simulator
from tests.project_toy import PROJECT, TOY_PATH
from tests.project_toy.experiments import (
    ToyDecay,
    ToyDecayHalf,
    ToyExperiment,
)
from tests.project_toy.records import adopt_toy_fit, toy_model_file


@pytest.fixture(scope="module")
def toy_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of the adopted fit of the toy project."""
    out = tmp_path_factory.mktemp("toy") / "out"
    adopt_toy_fit(out)
    return out


@pytest.fixture(scope="module")
def records(toy_out: Path) -> AdoptedRecords:
    """Get the records of the adopted fit."""
    (fit_id,) = json.loads((toy_out / "adopted.json").read_text(encoding="utf-8"))[
        "fits"
    ].values()
    return AdoptedRecords(toy_out / "fits", [fit_id])


@pytest.fixture(scope="module")
def toy_with_uncertainty(
    records: AdoptedRecords, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Create the toy model whose rate is the value of the fit with its uncertainty."""
    return toy_model_file(tmp_path_factory.mktemp("model") / "toy.xml", records)


def adopted(toy_out: Path, variants: dict[str, Path]) -> Project:
    """Get the toy project with the adopted fit and other variants."""
    return PROJECT.replace(
        fits_dir=toy_out / "fits",
        adopted=toy_out / "adopted.json",
        variants=variants,
        factory_dir=PROJECT.factory_dir,
    )


# ----------------------------------------------------------------------------
# the check of the fitted parameters of the variants
# ----------------------------------------------------------------------------
def test_the_check_of_the_fit_parameters(
    toy_out: Path, toy_with_uncertainty: Path, tmp_path: Path
) -> None:
    # the toy project has no adopted fits
    assert check_fit_parameters(PROJECT) == []
    project = adopted(toy_out, {"default": toy_with_uncertainty})
    assert check_fit_parameters(project) == []
    # the model of the factory has neither the value nor the distribution
    problems = check_fit_parameters(project.replace(variants=PROJECT.variants))
    assert any("has no distribution" in p for p in problems)
    assert any("take it from the record" in p for p in problems)
    # an adopted fit without a record
    file = tmp_path / "adopted.json"
    write_adopted(file, {"DECAY": "DECAY_20260101_000000__abcde"})
    (problem,) = check_fit_parameters(project.replace(adopted=file))
    assert "DECAY_20260101_000000__abcde" in problem


def test_the_check_of_the_fit_parameters_sees_every_variant(
    toy_out: Path,
    records: AdoptedRecords,
    toy_with_uncertainty: Path,
    tmp_path: Path,
) -> None:
    """Every variant has the fitted values and the distributions of the default one."""
    project = adopted(
        toy_out,
        {
            "default": toy_with_uncertainty,
            "other": toy_model_file(tmp_path / "other.xml", records, sid="other"),
        },
    )
    assert check_fit_parameters(project) == []
    stale = project.replace(
        variants={
            **project.variants,
            "other": toy_model_file(tmp_path / "stale.xml", None),
        }
    )
    problems = check_fit_parameters(stale)
    assert problems
    assert all("variant 'other', stale.xml" in p for p in problems)
    assert any("has no distribution" in p for p in problems)
    assert any("take it from the record" in p for p in problems)
    wider = project.replace(
        variants={
            **project.variants,
            "other": toy_model_file(tmp_path / "wider.xml", records, upper=5.0),
        }
    )
    (problem,) = check_fit_parameters(wider)
    assert "differs from the default variant" in problem
    missing = project.replace(
        variants={**project.variants, "other": tmp_path / "missing.xml"}
    )
    assert check_fit_parameters(missing) == [
        f"the variant 'other' {tmp_path / 'missing.xml'} does not exist"
    ]


# ----------------------------------------------------------------------------
# the settings and the draws
# ----------------------------------------------------------------------------
def test_the_settings_of_a_run() -> None:
    assert UncertaintySettings().quantiles == (0.05, 0.95)
    with pytest.raises(ValueError, match="two draws"):
        UncertaintySettings(draws=1)
    with pytest.raises(ValueError, match="quantiles"):
        UncertaintySettings(quantiles=(0.9, 0.1))


def test_the_draws_are_those_of_a_variant(
    toy_out: Path, records: AdoptedRecords, toy_with_uncertainty: Path, tmp_path: Path
) -> None:
    project = adopted(
        toy_out,
        {
            "default": toy_with_uncertainty,
            "plain": toy_model_file(tmp_path / "plain.xml", None),
        },
    )
    settings = UncertaintySettings(draws=4, seed=3)
    run_ = UncertaintyRun.create(project, settings)
    assert run_.id == UNCERTAINTY_DIMENSION
    assert run_.quantiles == (0.05, 0.95)
    expected = uncertainty_dimension(toy_with_uncertainty, 4, seed=3)
    assert np.array_equal(
        np.asarray(run_.dimension.values["k"]), np.asarray(expected.values["k"])
    )
    with pytest.raises(ValueError, match="No parameter"):
        UncertaintyRun.create(project, settings, variant="plain")


def draws_run(path: Path, draws: int = 3) -> UncertaintyRun:
    """Get a run over the draws of a model."""
    settings = UncertaintySettings(draws=draws)
    return UncertaintyRun(settings, uncertainty_dimension(path, draws, seed=1))


def initialized(
    experiment: type[ToyExperiment], run_: UncertaintyRun | None
) -> ToyExperiment:
    """Initialize an experiment, over the draws if given, as a run does."""
    settings: dict[str, Any] = {} if run_ is None else {"uncertainty": run_}
    runner = ExperimentRunner(
        experiment_classes=[experiment],
        base_path=TOY_PATH,
        simulator=None,
        on_error="raise",
        **settings,
    )
    (exp,) = runner.experiments.values()
    assert isinstance(exp, ToyExperiment)
    return exp


# ----------------------------------------------------------------------------
# the experiments over the draws
# ----------------------------------------------------------------------------
def test_an_experiment_runs_over_the_draws(toy_with_uncertainty: Path) -> None:
    run_ = draws_run(toy_with_uncertainty)
    exp = initialized(ToyDecay, run_)
    assert exp.uncertainty is run_
    assert exp.across() == UNCERTAINTY_DIMENSION
    simulation = exp._simulations["decay"]
    assert isinstance(simulation, Scan)
    assert [d.id for d in simulation.dimensions] == [UNCERTAINTY_DIMENSION]
    (plot,) = exp._figures["fig_decay"].get_plots()
    assert not [c for c in plot.curves if c.y.task_id is not None]
    (band,) = plot.bands
    assert isinstance(band, Band)
    assert band.across == UNCERTAINTY_DIMENSION
    assert band.name == "A"


def test_a_parameter_which_a_simulation_sets_is_not_drawn(
    toy_with_uncertainty: Path,
) -> None:
    exp = initialized(ToyDecayHalf, draws_run(toy_with_uncertainty))
    # the half rate sets the rate itself, nothing is left to draw: it runs
    # once and keeps its curve
    assert isinstance(exp._simulations["decay"], Simulation)
    (plot,) = exp._figures["fig_decay"].get_plots()
    assert len(plot.curves) == 1
    assert not plot.bands


def test_without_draws_the_experiment_is_unchanged() -> None:
    exp = initialized(ToyPhase, None)
    assert exp.uncertainty is None
    assert exp.across() is None
    assert isinstance(exp._simulations["decay"], Simulation)
    assert set(exp._simulations) == {"decay"}
    assert set(exp._tasks) == {"task_decay"}
    (plot,) = exp._figures["fig_decay"].get_plots()
    assert not plot.bands
    # the curves of the definition, as the experiment defines them
    (defined,) = ToyPhase(base_path=TOY_PATH).figures()["fig_decay"].get_plots()

    def lines(curves: list[Curve]) -> list[tuple[Any, ...]]:
        return [(c.name, c.x.index, c.y.index, c.y.task_id, c.y.sel) for c in curves]

    assert lines(plot.curves) == lines(defined.curves)
    assert len(plot.curves) == 2


def test_a_wrong_setting_raises() -> None:
    with pytest.raises(TypeError, match="no UncertaintyRun"):
        ExperimentRunner(
            experiment_classes=[ToyDecay],
            base_path=TOY_PATH,
            simulator=None,
            on_error="raise",
            uncertainty=3,
        )


def test_a_scan_gets_the_draws_appended(toy_with_uncertainty: Path) -> None:
    run_ = draws_run(toy_with_uncertainty)
    scan = Scan(
        Simulation(end=10, steps=10),
        [Dimension("dim_volume", values={"Vc": np.array([1.0, 2.0])})],
    )
    over = _over_draws(scan, run_.dimension, "scan", "exp")
    assert isinstance(over, Scan)
    assert [d.id for d in over.dimensions] == ["dim_volume", UNCERTAINTY_DIMENSION]
    clash = Scan(Simulation(end=10, steps=10), [run_.dimension])
    with pytest.raises(ValueError, match="id of the draws"):
        _over_draws(clash, run_.dimension, "scan", "exp")


def test_a_parameter_which_the_model_has_not_is_not_drawn(
    toy_with_uncertainty: Path,
) -> None:
    run_ = draws_run(toy_with_uncertainty)
    dimension = Dimension(
        UNCERTAINTY_DIMENSION,
        values={**run_.dimension.values, "missing": np.ones(3)},
    )
    over = _over_draws(Simulation(end=1, steps=1), dimension, "s", "e", {"k"})
    assert isinstance(over, Scan)
    assert set(over.dimensions[0].values) == {"k"}
    # a model with none of the drawn parameters runs the simulation once
    simulation = Simulation(end=1, steps=1)
    assert _over_draws(simulation, dimension, "s", "e", {"x"}) is simulation


class ToyPhase(ToyDecay):
    """The decay with a curve of A over A, whose x is not the time."""

    @override
    def figures(self) -> dict[str, Figure]:
        figures = super().figures()
        (plot,) = figures["fig_decay"].get_plots()
        plot.curves.append(
            Curve(
                x=Data(index="[A]", task="task_decay"),
                y=Data(index="[A]", task="task_decay"),
                name="A over A",
            )
        )
        return figures


def test_a_curve_over_another_variable_draws_the_fitted_values(
    toy_with_uncertainty: Path,
) -> None:
    exp = initialized(ToyPhase, draws_run(toy_with_uncertainty))
    (plot,) = exp._figures["fig_decay"].get_plots()
    (band,) = plot.bands
    assert band.y.task_id == "task_decay"
    (curve,) = [c for c in plot.curves if c.y.task_id == "task_decay_fitted"]
    assert curve.x.task_id == "task_decay_fitted"
    assert curve.name == "A over A (fit)"
    # below it a line of every draw, without a legend entry
    lines = [c for c in plot.curves if c.y.task_id == "task_decay"]
    assert plot.curves[: len(lines)] == lines
    assert [line.y.sel for line in lines] == [
        {UNCERTAINTY_DIMENSION: k} for k in range(3)
    ]
    assert len({line.y.sid for line in lines}) == 3
    assert all(line.name is None for line in lines)
    assert all(line.kwargs["alpha"] == 0.3 for line in lines)
    assert isinstance(exp._simulations["decay_fitted"], Simulation)
    assert exp._tasks["task_decay_fitted"].simulation_id == "decay_fitted"


class ToyInitialScan(ToyDecay):
    """The decay at three initial concentrations, A at its end over them."""

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:  # ty: ignore[invalid-method-override]
        return {
            **super().simulations(),
            "scan_initial": Scan(
                Simulation(end=10, steps=50),
                [Dimension("dim_initial", values={"[A]": Q([0.5, 1, 2], "mM")})],
            ),
        }

    @override
    def tasks(self) -> dict[str, Any]:
        from sbmlsim.task import Task

        return {
            **super().tasks(),
            "task_scan_initial": Task(model="model", simulation="scan_initial"),
        }

    @override
    def figures(self) -> dict[str, Figure]:
        figures = super().figures()
        (plot,) = figures["fig_decay"].get_plots()
        plot.curves.append(
            Curve(
                x=Data(index="dim_initial.[A]", task="task_scan_initial"),
                y=Data(index="[A]", task="task_scan_initial"),
                name="A over the initial A",
            )
        )
        return figures


def test_a_curve_over_a_dimension_of_its_scan_draws_a_band(
    toy_with_uncertainty: Path,
) -> None:
    """The values of a dimension are the same in every draw, the curve has a band."""
    exp = initialized(ToyInitialScan, draws_run(toy_with_uncertainty))
    (plot,) = exp._figures["fig_decay"].get_plots()
    assert not [c for c in plot.curves if c.y.task_id is not None]
    band = next(b for b in plot.bands if b.y.task_id == "task_scan_initial")
    assert band.x.index == "dim_initial.[A]"
    assert band.name == "A over the initial A"
    assert "scan_initial_fitted" not in exp._simulations


class ToyOnModel(ToyExperiment):
    """The decay on the model of the setting `model`."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": AbstractModel(self.settings["model"])}


def test_the_draws_reach_another_variant(
    records: AdoptedRecords, toy_with_uncertainty: Path, tmp_path: Path
) -> None:
    """The draws of the default variant run on another one, which has the same ids.

    Every draw of the rate decays from the start of the other variant:
    A(t) = 0.8 exp(-k t).
    """
    other = toy_model_file(tmp_path / "other.xml", records, initial=0.8)
    run_ = draws_run(toy_with_uncertainty)
    runner = ExperimentRunner(
        experiment_classes=[ToyOnModel],
        base_path=TOY_PATH,
        simulator=Simulator(n_workers=1),
        on_error="raise",
        uncertainty=run_,
        model=other,
    )
    (exp,) = runner.experiments.values()
    simulation = exp._simulations["decay"]
    assert isinstance(simulation, Scan)
    assert [d.id for d in simulation.dimensions] == [UNCERTAINTY_DIMENSION]
    exp.run(runner.simulator, output_path=None)
    result = exp.results["task_decay"]
    rates = np.asarray(run_.dimension.values["k"], dtype=float)
    assert len(set(rates)) == len(rates) == 3
    for index, k in enumerate(rates):
        times, values = point_timecourse(
            result, "[A]", **{UNCERTAINTY_DIMENSION: index}
        )
        np.testing.assert_allclose(
            np.asarray(values),
            0.8 * np.exp(-k * np.asarray(times)),
            rtol=1e-6,
            atol=1e-8,
        )


# ----------------------------------------------------------------------------
# the run of a project and the kind simulate
# ----------------------------------------------------------------------------
def test_a_project_runs_its_experiments_over_the_draws_in_a_pool(
    toy_out: Path, toy_with_uncertainty: Path, tmp_path: Path
) -> None:
    """The draws reach the workers of a pool, whose results are those of a serial run."""
    project = adopted(toy_out, {"default": toy_with_uncertainty})
    outs: dict[int, Path] = {}
    for n_workers in (2, 1):
        out = outs[n_workers] = tmp_path / f"workers_{n_workers}"
        results = project.run_experiments(
            ["ToyDecay", "ToyDecayHalf"],
            out,
            n_workers=n_workers,
            progress=False,
            figure_formats=["png"],
            save_results=True,
            uncertainty=UncertaintySettings(draws=3, seed=2),
            collect_figures=True,
        )
        assert not any(result.failed for result in results)
        assert (out / "index.html").exists()
        collected = sorted(p.name for p in (out / FIGURES_DIR).iterdir())
        assert collected == ["ToyDecayHalf_fig_decay.png", "ToyDecay_fig_decay.png"]
        # the parent keeps its experiments for the report, over the same draws
        decay = next(r.experiment for r in results if r.experiment.sid == "ToyDecay")
        assert decay.uncertainty is not None
        assert decay.uncertainty.settings.draws == 3
        assert isinstance(decay._simulations["decay"], Scan)
    files = sorted(p.relative_to(outs[1]) for p in outs[1].rglob("*.nc"))
    assert sorted(f.name for f in files) == [
        "ToyDecayHalf_task_decay.nc",
        "ToyDecay_task_decay.nc",
    ]
    for file in files:
        serial = ScanResult.from_netcdf(outs[1] / file)
        pooled = ScanResult.from_netcdf(outs[2] / file)
        xr.testing.assert_identical(pooled.ds, serial.ds)
    drawn = ScanResult.from_netcdf(outs[1] / "ToyDecay" / "ToyDecay_task_decay.nc")
    assert drawn.ds.sizes[UNCERTAINTY_DIMENSION] == 3


def test_the_figures_are_collected_before_the_failures_are_raised(
    tmp_path: Path,
) -> None:
    class ToyBroken(ToyDecay):
        variant = "missing"

    with pytest.raises(ExperimentRunError):
        PROJECT.run_experiments(
            [ToyDecay, ToyBroken],
            tmp_path,
            n_workers=1,
            progress=False,
            figure_formats=["svg"],
            collect_figures=True,
        )
    assert list((tmp_path / FIGURES_DIR).glob("ToyDecay_*.svg"))


def test_the_simulate_kind_runs_over_the_draws(
    toy_out: Path,
    toy_with_uncertainty: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.project_toy.PROJECT",
        adopted(toy_out, {"default": toy_with_uncertainty}),
    )
    spec = {
        "project": "tests.project_toy:PROJECT",
        "experiments": ["ToyDecay"],
        "uncertainty": {"draws": 3},
    }
    summary = run.execute("simulate", spec, tmp_path / "out", workers=1)
    assert summary["draws"] == 3
    assert summary["failed"] == 0
    planned = json.loads((tmp_path / "out" / "spec.json").read_text(encoding="utf-8"))
    assert planned["uncertainty"]["seed"] == summary["seed"]
    assert list((tmp_path / "out" / "ToyDecay").glob("*fig_decay*"))


def test_the_draws_of_the_simulate_kind_are_checked(tmp_path: Path) -> None:
    spec = {
        "project": "tests.project_toy:PROJECT",
        "uncertainty": {"draws": 1},
    }
    with pytest.raises((ValidationError, ConfigurationError), match="two draws"):
        run.execute("simulate", spec, tmp_path / "out", workers=1)
    # the model of the toy project has no distributions
    spec = {"project": "tests.project_toy:PROJECT", "uncertainty": {"draws": 3}}
    with pytest.raises(ConfigurationError, match="No parameter"):
        run.execute("simulate", spec, tmp_path / "other", workers=1)


def test_a_spec_of_draws_is_checked_without_the_experiments() -> None:
    """The run layer parses a spec of draws without importing the experiments."""
    code = (
        "import sys\n"
        "from sbmlsim.run.scan_kinds import SimulateSpec\n"
        "SimulateSpec(project='a:b', uncertainty={'draws': 3})\n"
        "try:\n"
        "    SimulateSpec(project='a:b', uncertainty={'quantiles': [0.9, 0.1]})\n"
        "except ValueError as err:\n"
        "    assert 'quantiles' in str(err)\n"
        "else:\n"
        "    raise AssertionError('not checked')\n"
        "assert 'sbmlsim.experiment' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
