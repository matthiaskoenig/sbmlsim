"""The subcommands of `python -m sbmlsim.run verify` on the toy project."""

import copy
import importlib.metadata
import json
import math
import pickle
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from PIL import Image

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.plot import Figure
from sbmlsim.project import FigureOutputs
from sbmlsim.simulation import Scan, Simulation
from sbmlsim.simulation.sampling import local
from sbmlsim.verify.cli import verify
from sbmlsim.verify.dump import Dump, figure_spec, load
from sbmlsim.verify.match import matched_rows
from sbmlsim.verify.renames import apply_renames, load_renames, rename_mapping
from tests.project_toy import PROJECT, TOY_PATH
from tests.project_toy.experiments import ToyDecay

MODEL_PATH = PROJECT.variant_path("default")
ROOT = Path(__file__).resolve().parents[2]
TOY = ["--project", "tests.project_toy:PROJECT", "--root", str(ROOT)]
BROKEN = ["--project", "tests.verify.projects:BROKEN", "--root", str(ROOT)]


def dump_args(out: Path, *more: str, project: list[str] = TOY) -> list[str]:
    return ["dump", *project, "--out", str(out), *more]


def test_dump_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--atol", "model", "--rtol", "1e-10"]
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    assert verify(dump_args(old, *common)) == 0
    assert verify(dump_args(new, "--order", str(old), *common)) == 0
    dumped = pickle.loads(new.read_bytes())
    assert dumped["results"]
    assert dumped["project"] == "toy"
    assert all(isinstance(v[1], np.ndarray) for v in dumped["results"].values())
    capsys.readouterr()
    assert verify(["compare", str(old), str(new)]) == 0
    match = re.search(r"simulations: (\d+) arrays compared", capsys.readouterr().out)
    assert match
    assert int(match.group(1)) == len(dumped["results"]) > 0


def test_a_simulation_on_a_grid() -> None:
    """The output times of `dump --grid`: equidistant, plus the times of the changes."""
    from sbmlsim import Q
    from sbmlsim.simulation import Change, Dimension
    from sbmlsim.verify.dump import on_grid

    simulation = Simulation(
        start=-24,
        end=12,
        time_unit="hr",
        preinit_changes={"A": 1.0},
        changes=[Change([0, 1.5], {"A": "A + 1"}), Change(Q(360, "min"), {"A": 0.0})],
        steps=10,
    )
    gridded = on_grid(simulation, 4)
    assert isinstance(gridded, Simulation)
    assert gridded.steps is None
    assert gridded.times == (-24.0, -15.0, -6.0, 0.0, 1.5, 3.0, 6.0, 12.0)
    assert (gridded.start, gridded.end, gridded.time_unit) == (-24, 12, "hr")
    assert gridded.preinit_changes == simulation.preinit_changes
    assert gridded.changes == simulation.changes
    scan = Scan(
        simulation=Simulation(end=10),
        dimensions=[Dimension("dim_a", values={"A": [1.0, 2.0]}, at=2.5)],
    )
    gridded_scan = on_grid(scan, 2)
    assert isinstance(gridded_scan, Scan)
    assert gridded_scan.simulation.times == (0.0, 2.5, 5.0, 10.0)
    assert gridded_scan.dimensions == scan.dimensions


def test_the_simulations_of_a_dimension_on_a_grid() -> None:
    """Every simulation of a dimension of simulations gets its own grid."""
    from sbmlsim import Q
    from sbmlsim.simulation import Change, Dimension
    from sbmlsim.verify.dump import on_grid

    short = Simulation(end=4, steps=100, changes=[Change(1.5, {"A": 1.0})])
    long = Simulation(start=-2, end=8)
    scan = Scan(
        simulation=Simulation(end=10),
        dimensions=[Dimension("dim_sim", simulations={"short": short, "long": long})],
    )
    gridded = on_grid(scan, 2)
    assert isinstance(gridded, Scan)
    assert gridded.simulation.times == (0.0, 5.0, 10.0)
    (dimension,) = gridded.dimensions
    assert dimension.id == "dim_sim"
    assert dimension.simulations["short"].times == (0.0, 1.5, 2.0, 4.0)
    assert dimension.simulations["short"].steps is None
    assert dimension.simulations["long"].times == (-2.0, 3.0, 8.0)
    # a quantity of a simulation without a time unit has no number on the grid
    unknown = Simulation(end=4, changes=[Change(Q(1, "hr"), {"A": 1.0})])
    with pytest.raises(ValueError, match="no time unit"):
        on_grid(unknown, 2)
    with pytest.raises(ValueError, match="no simulation or scan"):
        on_grid("task", 2)  # ty: ignore[invalid-argument-type]


def test_dumps_on_a_common_grid_compare_their_times_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two dumps at other tolerances share their output times with `--grid`."""
    dumps = []
    for rtol in ("1e-10", "1e-9"):
        out = tmp_path / f"grid_{rtol}.pkl"
        args = dump_args(out, "--rtol", rtol, "--grid", "20", "ToyDecay")
        assert verify(args) == 0
        dumps.append(str(out))
    dumped = pickle.loads(Path(dumps[0]).read_bytes())
    assert dumped["grid"] == 20
    (times,) = [v[1] for k, v in dumped["results"].items() if k[2] == "time"]
    assert np.allclose(times.ravel(), np.linspace(0, 10, 21))
    capsys.readouterr()
    assert verify(["compare", *dumps]) == 0
    out = capsys.readouterr().out
    assert "0 unresolved on this grid" in out
    assert "rows at other times: 0 rows of the old dump and 0 of the new dump" in out
    assert "grid: old 20, new 20" in out


def test_definitions(tmp_path: Path) -> None:
    out = tmp_path / "definitions.json"
    assert verify(["definitions", *TOY, "--out", str(out)]) == 0
    content = json.loads(out.read_text(encoding="utf-8"))
    assert set(content) == {"DECAY"}
    assert [p["pid"] for p in content["DECAY"]["parameters"]] == ["k"]
    # the checkout is replaced
    assert str(ROOT) not in json.dumps(content)


def test_fits_run_a_minimal_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert verify(["fits", *TOY, "--out", str(tmp_path / "fits")]) == 0
    assert "'DECAY': 'ok'" in capsys.readouterr().out
    assert verify(["fits", *TOY, "--out", str(tmp_path), "OTHER"]) == 2


def test_sbml(tmp_path: Path) -> None:
    copy_ = tmp_path / "copy.xml"
    shutil.copy(MODEL_PATH, copy_)
    assert verify(["sbml", str(MODEL_PATH), str(copy_)]) == 0
    changed = tmp_path / "changed.xml"
    changed.write_text(
        MODEL_PATH.read_text(encoding="utf-8").replace("0.2", "0.3"),
        encoding="utf-8",
    )
    assert verify(["sbml", str(MODEL_PATH), str(changed)]) == 1


def write_png(path: Path, value: int = 0) -> None:
    """Write a small image, a different one for a different value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_png(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "a"
    PROJECT.run_experiments(
        ["studies"], output, n_workers=1, figure_formats=["png"], progress=False
    )
    figures = list(output.rglob("*.png"))
    assert figures
    # the figures are below the experiments, not in the directory itself
    assert not list(output.glob("*.png"))
    capsys.readouterr()
    assert verify(["png", str(output), str(output)]) == 0
    assert f"{len(figures)} identical, 0 differ" in capsys.readouterr().out


def test_png_compares_recursively_by_relative_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for directory in (old, new):
        write_png(directory / "A" / "fig.png")
        write_png(directory / "_figures" / "fig.png", 7)
    capsys.readouterr()
    assert verify(["png", str(old), str(new)]) == 0
    assert "2 identical, 0 differ" in capsys.readouterr().out
    write_png(new / "A" / "fig.png", 200)
    assert verify(["png", str(old), str(new)]) == 1
    assert "differs: A" in capsys.readouterr().out


def test_png_reports_files_on_one_side(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    write_png(old / "A" / "fig.png")
    write_png(new / "A" / "fig.png")
    write_png(old / "B" / "only_old.png")
    write_png(new / "C" / "only_new.png")
    capsys.readouterr()
    assert verify(["png", str(old), str(new)]) == 1
    out = capsys.readouterr().out
    assert "only in old: B" in out
    assert "only in new: C" in out
    assert "1 identical, 2 differ" in out


def test_png_without_figures_or_directory_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert verify(["png", str(empty), str(empty)]) == 2
    assert verify(["png", str(tmp_path / "missing"), str(empty)]) == 2
    assert verify(["png", str(empty), str(tmp_path / "missing")]) == 2


def test_sbml_compares_directories_recursively(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for directory in (old, new):
        (directory / "sub").mkdir(parents=True)
        shutil.copy(MODEL_PATH, directory / "sub" / "toy.xml")
    capsys.readouterr()
    assert verify(["sbml", str(old), str(new)]) == 0
    assert "sub" in capsys.readouterr().out
    (new / "sub" / "toy.xml").write_text(
        MODEL_PATH.read_text(encoding="utf-8").replace("0.2", "0.3"),
        encoding="utf-8",
    )
    assert verify(["sbml", str(old), str(new)]) == 1
    shutil.copy(MODEL_PATH, new / "extra.xml")
    capsys.readouterr()
    assert verify(["sbml", str(old), str(new)]) == 1
    assert "extra.xml: only in new" in capsys.readouterr().out


def test_sbml_without_files_or_directory_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert verify(["sbml", str(empty), str(empty)]) == 2
    assert verify(["sbml", str(tmp_path / "missing"), str(empty)]) == 2


def test_a_project_from_another_root_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    out = tmp_path / "definitions.json"
    code = verify(
        [
            "definitions",
            "--project",
            "tests.project_toy:PROJECT",
            "--root",
            str(other),
            "--out",
            str(out),
        ]
    )
    assert code == 4
    assert "not from below" in caplog.text
    assert not out.exists()


def test_an_unknown_project_is_a_usage_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = str(tmp_path / "x.pkl")
    assert verify(["dump", "--project", "tests.project_toy:MISSING", "--out", out]) == 4
    assert "no attribute 'MISSING'" in caplog.text
    assert verify(["dump", "--project", "no-reference", "--out", out]) == 4
    assert verify(["dump", "--out", out]) == 2


def test_the_report_writes_the_text_as_it_is(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sbmlsim.verify.common import report

    report("a\tb :smile: [bold]x[/bold]")
    assert capsys.readouterr().out == "a\tb :smile: [bold]x[/bold]\n"


def test_the_versions_of_a_dump() -> None:
    from sbmlsim.verify.dump import package_versions

    versions = package_versions(PROJECT)
    assert versions["sbmlsim"] == importlib.metadata.version("sbmlsim")
    assert "sbmlutils" in versions  # a core library of the release settings


def test_the_command_is_part_of_the_run_layer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from sbmlsim.run.cli import main

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["verify", "png", str(empty), str(empty)]) == 2
    assert main(["verify", "--help"]) == 0
    assert "dump" in capsys.readouterr().out
    assert main(["verify", "--profile", "png", str(empty), str(empty)]) == 2


def test_capture_settings_of_a_project() -> None:
    from sbmlsim.verify.definitions import capture_settings

    settings = capture_settings(PROJECT, ToyDecay, ROOT)
    assert settings["experiment"] == "ToyDecay"
    assert settings["simulator"]["relative_tolerance"] == 1e-10
    assert settings["run"]["n_workers"] is None
    assert settings["run"]["figure_formats"] == ["svg", "png"]
    assert settings["run"]["reduced_selections"] is True
    assert settings["run"]["save_results"] is False
    assert settings["rc"]["savefig.dpi"] == 30
    assert settings["runner"]["jit_optimization"] == "O0"
    assert settings["calls"] == {"simulator": 1, "runner": 1, "run": 1}
    assert "<root>" in settings["runner"]["base_path"]
    assert "figure" not in settings


def _fake_run(positional: bool):
    from sbmlsim.experiment import ExperimentRunner
    from sbmlsim.simulator import Simulator

    def run(experiment: type[SimulationExperiment], output_dir: Path) -> None:
        if positional:
            simulator = Simulator(1, absolute_tolerance=1e-10)
            runner = ExperimentRunner([experiment], TOY_PATH, None, simulator)
            runner.run_experiments(output_dir, False, False, ["png"])
        else:
            simulator = Simulator(n_workers=1, absolute_tolerance=1e-10)
            runner = ExperimentRunner(
                experiment_classes=[experiment],
                simulator=simulator,
                base_path=TOY_PATH,
            )
            runner.run_experiments(
                output_path=output_dir,
                show_figures=False,
                save_results=False,
                figure_formats=["png"],
            )

    return run


def test_capture_settings_binds_positional_arguments() -> None:
    from sbmlsim.verify.definitions import capture_settings

    a = capture_settings(PROJECT, ToyDecay, ROOT, _fake_run(True))
    b = capture_settings(PROJECT, ToyDecay, ROOT, _fake_run(False))
    assert a == b
    assert a["simulator"]["absolute_tolerance"] == 1e-10
    assert a["run"]["figure_formats"] == ["png"]
    assert a["runner"]["experiment_classes"] == ["ToyDecay"]


def test_capture_settings_requires_a_complete_run() -> None:
    from sbmlsim.verify.definitions import capture_settings

    with pytest.raises(RuntimeError, match="did not reach sbmlsim"):
        capture_settings(PROJECT, ToyDecay, ROOT, lambda experiment, out: None)


def test_capture_settings_records_the_matplotlib_settings() -> None:
    import matplotlib

    from sbmlsim.verify.definitions import capture_settings

    def with_font(experiment: type[SimulationExperiment], output_dir: Path) -> None:
        with matplotlib.rc_context({"font.size": 3}):
            PROJECT.run_experiments(experiment, output_dir)

    rc = capture_settings(PROJECT, ToyDecay, ROOT, with_font)["rc"]
    assert rc["font.size"] == 3
    assert rc["savefig.dpi"] == 30


def test_capture_settings_sees_a_difference() -> None:
    from sbmlsim.verify.definitions import capture_settings

    other = PROJECT.replace(figures=FigureOutputs(formats=("png",), savefig_dpi=55))
    a = capture_settings(PROJECT, ToyDecay, ROOT)
    b = capture_settings(other, ToyDecay, ROOT)
    assert a["run"]["figure_formats"] != b["run"]["figure_formats"]
    assert b["rc"]["savefig.dpi"] == 55


def test_settings_writes_a_file(tmp_path: Path) -> None:
    out = tmp_path / "settings.json"
    assert verify(["settings", *TOY, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["experiment"] == "ToyDecay"
    assert verify(["settings", *TOY, "--out", str(out), "--experiment", "No"]) == 2


def test_settings_of_a_run_which_misses_sbmlsim_exit_with_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(type(PROJECT), "run_experiments", lambda self, *a, **k: None)
    assert verify(["settings", *TOY, "--out", str(tmp_path / "s.json")]) == 2


class ToyDesign(ToyDecay):
    """The decay with a local design of k, whose creation constructs simulators."""

    def simulations(self) -> dict[str, Simulation | Scan]:  # ty: ignore[incompatible-method-override]
        """The decay and its local design."""
        sim = Simulation(end=10, steps=50)
        return {
            "decay": sim,
            "decay_sensitivity": Scan(
                sim,
                [local(["k"], delta=0.5, model=self._models["model"], simulation=sim)],
            ),
        }


def test_capture_settings_of_an_experiment_with_a_design() -> None:
    from sbmlsim.verify.definitions import capture_settings

    settings = capture_settings(PROJECT, ToyDesign, ROOT)
    assert settings["simulator"]["n_workers"] is None
    assert settings["calls"]["simulator"] == 1


def test_plain_converts_nested_paths() -> None:
    from sbmlsim.verify.definitions import _plain

    value = _plain({"a": [{"b": ROOT / "x"}], "c": (ROOT,)})
    assert value == {"a": [{"b": (ROOT / "x").as_posix()}], "c": [ROOT.as_posix()]}


def test_dump_with_the_tolerances_of_the_project(tmp_path: Path) -> None:
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out, "--atol", "model", "--rtol", "model")) == 0


def test_dump_with_a_bad_tolerance_exits_with_2(tmp_path: Path) -> None:
    for option in ("--atol", "--rtol"):
        assert verify(dump_args(tmp_path / "d.pkl", option, "abc")) == 2


def test_a_dump_of_unknown_experiments_exits_with_2(tmp_path: Path) -> None:
    assert verify(dump_args(tmp_path / "d.pkl", "Missing")) == 2


def test_a_dump_records_an_experiment_which_cannot_be_initialized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # the experiment which cannot be initialized runs first
    order = tmp_path / "order.pkl"
    with order.open("wb") as f:
        pickle.dump({"order": ["ToyCannotInitialize", "ToyDecay", "ToyFigureFails"]}, f)
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out, "--order", str(order), project=BROKEN)) == 1
    dump = load(str(out))
    assert dump["order"] == ["ToyCannotInitialize", "ToyDecay", "ToyFigureFails"]
    assert list(dump["errors"]) == ["ToyCannotInitialize"]
    assert (
        "RuntimeError: the simulations of ToyCannotInitialize fail"
        in dump["errors"]["ToyCannotInitialize"]
    )
    # the experiments after it ran and are stored
    assert {sid for sid, _, _ in dump["results"]} == {"ToyDecay", "ToyFigureFails"}
    assert ("ToyDecay", "task_decay", "[A]") in dump["results"]
    assert set(dump["mappings"]) == {"ToyDecay", "ToyFigureFails"}


def test_a_dump_stores_the_time_of_the_grid(tmp_path: Path) -> None:
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out)) == 0
    dims, values = load(str(out))["results"]["ToyDecay", "task_decay", "time"]
    assert dims == ("time",)
    assert values[0] == 0.0
    assert values.size == 51
    dims, _ = load(str(out))["results"]["ToyDecay", "task_decay", "[A]"]
    assert dims == ("time",)


def test_a_dump_names_the_versions(tmp_path: Path) -> None:
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out, "ToyDecay")) == 0
    versions = load(str(out))["versions"]
    assert versions["sbmlsim"] == importlib.metadata.version("sbmlsim")


def test_a_dump_stores_the_settings_and_the_size_of_a_figure(tmp_path: Path) -> None:
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out, "ToyDecay")) == 0
    spec = json.loads(load(str(out))["figures"]["ToyDecay"]["fig_decay"])
    assert spec["__resolved__"]["settings"] == {}
    assert spec["__resolved__"]["width"] > 0


def test_the_settings_and_the_size_a_figure_is_rendered_with() -> None:
    class Wide(ToyDecay):
        figure_settings = MappingProxyType(
            {"panel_width": 3.0, "legend_position": "outside"}
        )

    experiment = Wide(base_path=TOY_PATH)
    figure = Figure(
        experiment, sid="f", num_cols=2, settings={"legend_position": "inside"}
    )
    resolved = json.loads(figure_spec(figure, experiment))["__resolved__"]
    assert resolved["settings"] == {"panel_width": 3.0, "legend_position": "inside"}
    assert resolved["width"] == 6.0


def test_the_figure_fields_are_left_out_at_the_level_of_the_figure(
    tmp_path: Path,
) -> None:
    def spec(width: float, extra: int | None = None) -> str:
        figure: dict[str, object] = {
            "sid": "f",
            "width": width,
            "subplots": [{"plot": {"width": width}}],
        }
        if extra is not None:
            figure["extra"] = extra
        return json.dumps(figure, sort_keys=True)

    def compared(old: str, new: str, *args: str) -> int:
        return verify(
            [
                "compare",
                written(tmp_path, "o.pkl", synthetic(figures={"E": {"f": old}})),
                written(tmp_path, "n.pkl", synthetic(figures={"E": {"f": new}})),
                *args,
            ]
        )

    renames = tmp_path / "renames.json"
    renames.write_text(json.dumps({"figure_fields": ["width"]}), encoding="utf-8")
    # the width of the figure is left out, the one of its plot is not
    assert compared(spec(8.0), spec(9.0)) == 1
    assert compared(spec(8.0), spec(8.0, extra=1)) == 1
    renames.write_text(json.dumps({"figure_fields": ["extra"]}), encoding="utf-8")
    assert compared(spec(8.0), spec(8.0, extra=1), "--renames", str(renames)) == 0
    renames.write_text(json.dumps({"figure_fields": ["width"]}), encoding="utf-8")
    assert compared(spec(8.0), spec(9.0), "--renames", str(renames)) == 1


def test_a_figure_field_of_one_dump_only_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_spec = {"sid": "f", "plots": [{"x": 1}]}
    new_spec = {"sid": "f", "width": None, "plots": [{"x": 1, "_bands": []}]}
    old = written(
        tmp_path, "o.pkl", synthetic(figures={"E": {"f": json.dumps(old_spec)}})
    )
    new = written(
        tmp_path, "n.pkl", synthetic(figures={"E": {"f": json.dumps(new_spec)}})
    )
    capsys.readouterr()
    assert verify(["compare", old, new]) == 1
    out = capsys.readouterr().out
    # a field which is null in one dump and missing in the other is a difference
    assert "width only in new: None" in out
    assert "plots[0]._bands only in new: []" in out


# --- the variants -----------------------------------------------------------------


def test_a_dump_records_the_variant_of_every_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A study which moves to the build of another variant is a difference."""
    old = tmp_path / "old.pkl"
    assert verify(dump_args(old)) == 0
    dumped = load(str(old))
    assert dumped["builds"] == {
        "ToyDecay": {"task_decay": "default"},
        "ToyDecaySlow": {"task_decay": "slow"},
        "ToyDecayHalf": {"task_decay": "default"},
    }
    capsys.readouterr()
    assert verify(["compare", str(old), str(old)]) == 0
    assert "builds: 3 old, 3 new, 0 differ" in capsys.readouterr().out

    moved = copy.deepcopy(dumped)
    moved["builds"]["ToyDecaySlow"]["task_decay"] = "default"
    new = written(tmp_path, "new.pkl", moved)
    assert verify(["compare", str(old), new]) == 1
    out = capsys.readouterr().out
    assert "builds: 3 old, 3 new, 1 differ (0 expected)" in out
    assert "ToyDecaySlow/task_decay" in out
    assert "'slow' -> 'default'" in out
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"builds": ["ToyDecaySlow/*"]}}), encoding="utf-8"
    )
    assert verify(["compare", str(old), new, "--renames", str(renames)]) == 0


def test_the_experiment_names_the_variant_of_a_task(tmp_path: Path) -> None:
    """The hook `SimulationExperiment.variant_of` wins over the model file."""
    from sbmlsim.task import Task
    from sbmlsim.verify.dump import variant_of

    class Named(ToyDecay):
        def variant_of(self, simulation: str) -> str | None:
            return f"named_{simulation}"

    experiment = Named(base_path=TOY_PATH)
    experiment._tasks = {"task_decay": Task("model", "decay")}
    assert variant_of(PROJECT, experiment, "task_decay") == "named_decay"
    plain = ToyDecay(base_path=TOY_PATH)
    assert plain.variant_of("decay") is None


def test_a_model_which_is_no_variant_has_no_build() -> None:
    from sbmlsim.model import AbstractModel
    from sbmlsim.task import Task
    from sbmlsim.verify.dump import variant_of

    other = PROJECT.replace(variants={"other": Path("models/toy_slow.xml")})
    experiment = ToyDecay(base_path=TOY_PATH)
    experiment._models = {"model": AbstractModel(MODEL_PATH)}  # ty: ignore[invalid-assignment]
    experiment._tasks = {"task_decay": Task("model", "decay")}
    assert variant_of(other, experiment, "task_decay") is None
    assert variant_of(PROJECT, experiment, "task_decay") == "default"


def test_a_variant_which_the_project_does_not_have_is_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sbmlsim.task import Task
    from sbmlsim.verify.dump import variant_of

    class Named(ToyDecay):
        def variant_of(self, simulation: str) -> str | None:
            return "nowhere"

    experiment = Named(base_path=TOY_PATH)
    experiment._tasks = {"task_decay": Task("model", "decay")}
    assert variant_of(PROJECT, experiment, "task_decay") == "nowhere"
    assert "no variant of the project" in caplog.text


def test_the_tasks_of_the_builds_are_renamed() -> None:
    dump = synthetic(builds={"E": {"task_a": "slow"}})
    renamed = apply_renames(dump, {"ids": {"task_a": "task_b"}})
    assert renamed["builds"] == {"E": {"task_b": "slow"}}


def test_a_dump_without_builds_is_compared_without_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = written(tmp_path, "old.pkl", synthetic())
    new = written(tmp_path, "new.pkl", synthetic(builds={"E": {"task": "slow"}}))
    assert verify(["compare", old, new]) == 0
    assert "builds: in the new dump only, not compared" in capsys.readouterr().out


def test_a_dump_and_the_fits_load_at_the_level_of_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`jit_optimization` of the project reaches the runner of a dump and the fits."""
    import sbmlsim.experiment
    import sbmlsim.fit.cli

    levels: list[object] = []
    runner_init = sbmlsim.experiment.ExperimentRunner.__init__

    def recording_init(self: object, *args: object, **kwargs: object) -> None:
        levels.append(kwargs.get("jit_optimization"))
        runner_init(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(sbmlsim.experiment.ExperimentRunner, "__init__", recording_init)
    assert verify(dump_args(tmp_path / "d.pkl", "ToyDecay")) == 0
    assert levels == ["O0"]

    seen: list[str] = []

    def recording_run_fit(definition: object, **kwargs: object) -> dict:
        seen.append(str(definition.problem(opid="level").jit_optimization))  # ty: ignore[missing-attribute]
        return {}

    monkeypatch.setattr(sbmlsim.fit.cli, "run_fit", recording_run_fit)
    assert verify(["fits", *TOY, "--out", str(tmp_path / "fits")]) == 0
    assert seen == ["O0"]


def test_the_experiments_of_the_modules_are_dumped_with_all(tmp_path: Path) -> None:
    out = tmp_path / "dump.pkl"
    assert verify(dump_args(out, "--all")) == 0
    assert "ToyUnregistered" in load(str(out))["order"]


def test_the_metadata_of_a_mapping_is_renamed() -> None:
    description = (
        "fm|d|t|time|A|4|None|None|t|t|time|[A]|None|None|None|weight=None|"
        + str([("data", "MEAN"), ("dosing", "MULTI"), ("route", "IV")])
    )
    renamed = rename_mapping(
        description,
        {
            "fields": {"data": "data_type"},
            "members": {"dosing": {"MULTI": "MULTIPLE"}},
            "added": {"application_form": "NR"},
        },
    )
    assert renamed.endswith(
        str(
            [
                ("application_form", "NR"),
                ("data_type", "MEAN"),
                ("dosing", "MULTIPLE"),
                ("route", "IV"),
            ]
        )
    )


def test_compare_applies_the_expected_renames(tmp_path: Path) -> None:
    old, new = tmp_path / "old.pkl", tmp_path / "new.pkl"
    assert verify(dump_args(old)) == 0
    dump = load(str(old))
    # the toy mappings have no metadata: give them a route
    dump["mappings"] = {
        sid: [d[: -len("[]")] + "[('route', 'IV')]" for d in descriptions]
        for sid, descriptions in dump["mappings"].items()
    }
    old.write_bytes(pickle.dumps(dump))
    dump["mappings"] = {
        sid: [d.replace("('route', 'IV')", "('route', 'IVX')") for d in descriptions]
        for sid, descriptions in dump["mappings"].items()
    }
    dump["figures"] = {
        sid: {key: spec.replace('"A"', '"A renamed"') for key, spec in figures.items()}
        for sid, figures in dump["figures"].items()
    }
    new.write_bytes(pickle.dumps(dump))
    assert dump["mappings"]["ToyDecay"]
    renames = tmp_path / "renames.json"
    assert verify(["compare", str(old), str(new)]) == 1
    renames.write_text(
        json.dumps(
            {
                "metadata": {"members": {"route": {"IV": "IVX"}}},
                "strings": {'"A"': '"A renamed"'},
            }
        ),
        encoding="utf-8",
    )
    assert verify(["compare", str(old), str(new), "--renames", str(renames)]) == 0


def test_a_renames_file_with_unknown_parts_is_an_error(tmp_path: Path) -> None:
    renames = tmp_path / "renames.json"
    renames.write_text(json.dumps({"metadata": {"member": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="member"):
        load_renames(str(renames))


@pytest.mark.parametrize(
    ("content", "part"),
    [
        ([], "object"),
        ({"strings": ["a"]}, "strings"),
        ({"strings": {"a": 1}}, "strings"),
        ({"metadata": []}, "metadata"),
        ({"metadata": {"fields": []}}, "fields"),
        ({"metadata": {"fields": {"a": 1}}}, "fields"),
        ({"metadata": {"members": {"route": "PO"}}}, "members"),
        ({"metadata": {"members": {"route": {"a": 1}}}}, "members"),
        ({"metadata": {"added": {"a": 1}}}, "added"),
    ],
)
def test_a_renames_file_of_the_wrong_type_is_an_error(
    tmp_path: Path, content: object, part: str
) -> None:
    renames = tmp_path / "renames.json"
    renames.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match=part):
        load_renames(str(renames))


def test_renamed_mappings_are_sorted_like_a_dump() -> None:
    old = {
        "mappings": {
            "E": [
                "fm|a|t|time|A|4|None|None|t|t|time|[A]|None|None|None|weight=None|"
                + str([("route", "B")]),
                "fm|a|t|time|A|4|None|None|t|t|time|[A]|None|None|None|weight=None|"
                + str([("route", "C")]),
            ]
        }
    }
    renames = {"metadata": {"members": {"route": {"B": "Z"}}}}
    renamed = apply_renames(old, renames)  # ty: ignore[invalid-argument-type]
    assert renamed["mappings"]["E"] == sorted(renamed["mappings"]["E"])
    assert renamed["mappings"]["E"][0].endswith(str([("route", "C")]))


T_OLD = np.array([0.0, 1.0, 2.0, 3.0])
#: a change at 2: the state before it, then the state after it
T_NEW = np.array([0.0, 1.0, 2.0, 2.0, 3.0])


def synthetic(
    results: dict | None = None,
    *,
    datasets: dict | None = None,
    mappings: dict | None = None,
    figures: dict | None = None,
    ranges: dict | None = None,
    versions: dict[str, str] | None = None,
    builds: dict[str, dict[str, str]] | None = None,
) -> Dump:
    dump = Dump(
        project="m",
        order=["E"],
        errors={},
        results=results or {},
        datasets=datasets or {},
        mappings=mappings or {"E": []},
        figures=figures or {"E": {}},
        ranges=ranges or {"E": {}},
    )
    if versions is not None:
        dump["versions"] = versions
    if builds is not None:
        dump["builds"] = builds
    return dump


def written(tmp_path: Path, name: str, dump: Dump) -> str:
    path = tmp_path / name
    path.write_bytes(pickle.dumps(dump))
    return str(path)


def old_scan() -> Dump:
    """A scan with the points first, (_point, dim_k), the state after the change at 2."""
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.5, 35.0], [4.0, 40.0]])
    return synthetic(
        {
            ("E", "task_x", "time"): (
                ("_point", "dim_k"),
                np.stack([T_OLD, T_OLD], axis=1),
            ),
            ("E", "task_x", "[A]"): (("_point", "dim_k"), values),
        }
    )


def new_scan(before: float = 3.0) -> Dump:
    """A scan: (dim_k, time), the states before and after the change."""
    values = np.array([[1.0, 2.0, before, 3.5, 4.0], [10.0, 20.0, 30.0, 35.0, 40.0]])
    return synthetic(
        {
            ("E", "task_x", "time"): (("time",), T_NEW),
            ("E", "task_x", "[A]"): (("dim_k", "time"), values),
        },
        versions={"sbmlsim": "0.9.3", "toy": "0.5.0"},
    )


def test_a_dump_without_the_states_before_a_change_compares_the_states_after(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Where one dump has fewer rows at a change, the state after it is compared."""
    values = np.array([[1.0, 2.0, 3.5, 4.0], [10.0, 20.0, 35.0, 40.0]])
    fewer = synthetic(
        {
            ("E", "task_x", "time"): (("time",), T_OLD),
            ("E", "task_x", "[A]"): (("dim_k", "time"), values),
        }
    )
    old, new = (
        written(tmp_path, "old.pkl", fewer),
        written(tmp_path, "new.pkl", new_scan()),
    )
    code = verify(["compare", old, new, "--threshold", "1e-12"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (
        "rows at a change: 1 rows of the new dump and 0 of the old dump left out" in out
    )
    assert "changes: E 1 rows" in out
    assert "new: sbmlsim 0.9.3, toy 0.5.0" in out


def test_the_matched_rows() -> None:
    match = matched_rows(T_OLD, T_NEW)
    assert match.old.tolist() == [0, 1, 2, 3]
    assert match.new.tolist() == [0, 1, 3, 4]
    assert (match.ignored_old, match.ignored_new, match.unmatched) == (0, 1, 0)
    same = matched_rows(T_NEW, T_NEW)
    assert same.new.tolist() == [0, 1, 2, 3, 4]
    assert (same.ignored_old, same.ignored_new) == (0, 0)


def test_two_dumps_compare_every_row(tmp_path: Path) -> None:
    a = written(tmp_path, "a.pkl", new_scan())
    b = written(tmp_path, "b.pkl", new_scan(before=3.1))
    assert verify(["compare", a, a]) == 0
    assert verify(["compare", a, b, "--threshold", "1e-12"]) == 1


def ragged(times: Sequence[Sequence[float]], values: Sequence[Sequence[float]]) -> Dump:
    """A scan in the ragged layout: every point has its own times.

    The rows of a point are padded with NaN to the longest point, `(dim_k, _point)`.
    """
    rows = max(len(t) for t in times)

    def padded(columns: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array([[*c, *[np.nan] * (rows - len(c))] for c in columns])

    return synthetic(
        {
            ("E", "task_x", "time"): (("dim_k", "_point"), padded(times)),
            ("E", "task_x", "[A]"): (("dim_k", "_point"), padded(values)),
        },
        versions={"sbmlsim": "0.9.3", "toy": "0.5.0"},
    )


def test_the_rows_of_every_point_are_matched_by_its_own_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ragged result: the second point has a change at 2, the first has not."""
    old = ragged([[0, 1, 2, 3], [0, 1, 2, 3]], [[1, 2, 3, 4], [10, 20, 35, 40]])
    new = ragged([[0, 1, 2, 3], [0, 1, 2, 2, 3]], [[1, 2, 3, 4], [10, 20, 30, 35, 40]])
    code = verify(
        [
            "compare",
            written(tmp_path, "old.pkl", old),
            written(tmp_path, "new.pkl", new),
            "--threshold",
            "1e-12",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "changes: E 1 rows" in out
    # the value after the change of the second point is compared
    new["results"]["E", "task_x", "[A]"][1][1, 3] = 36.0
    code = verify(
        [
            "compare",
            written(tmp_path, "old.pkl", old),
            written(tmp_path, "new.pkl", new),
            "--threshold",
            "1e-12",
        ]
    )
    assert code == 1


def test_a_time_of_another_point_than_the_first_is_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = ragged([[0, 1, 2, 3], [0, 1, 2, 3]], [[1, 2, 3, 4], [10, 20, 30, 40]])
    new = ragged([[0, 1, 2, 3], [0, 1, 2, 5]], [[1, 2, 3, 4], [10, 20, 30, 40]])
    code = verify(
        [
            "compare",
            written(tmp_path, "old.pkl", old),
            written(tmp_path, "new.pkl", new),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1, out
    # the old row at 3 is compared with the new rows at 2 and 5 interpolated,
    # the new row at 5 has no time of the old dump around it
    assert "rows at other times: 1 rows of the old dump and 0 of the new" in out
    assert "1 rows of the new dump without a time in the old dump" in out
    assert "times of E/task_x: 1 rows only in new, at 5" in out


def test_an_old_row_without_a_new_time_is_a_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = old_scan()
    times = T_OLD.copy()
    # after the last time of the new dump: no interpolation
    times[3] = 3.5
    old["results"]["E", "task_x", "time"] = (
        ("_point", "dim_k"),
        np.stack([times, times], axis=1),
    )
    code = verify(
        [
            "compare",
            written(tmp_path, "old.pkl", old),
            written(tmp_path, "new.pkl", new_scan()),
        ]
    )
    assert code == 1
    assert (
        "1 rows of the old dump without a time in the new dump"
        in capsys.readouterr().out
    )


def timecourse(
    times: Sequence[float] | np.ndarray, values: Sequence[float] | np.ndarray
) -> Dump:
    """A dump of one task with its output times and the values of A."""
    return synthetic(
        {
            ("E", "t", "time"): (("time",), np.array(times)),
            ("E", "t", "[A]"): (("time",), np.array(values)),
        }
    )


def test_repeated_times_are_matched_from_the_last_row() -> None:
    # two dumps with other output times: every row of the
    # common times is compared, the states before a change included
    match = matched_rows(np.array([0.0, 1, 2, 2, 3]), np.array([0.0, 1, 2, 2, 3, 4]))
    assert match.old.tolist() == [0, 1, 2, 3, 4]
    assert match.new.tolist() == [0, 1, 2, 3, 4]
    assert match[2:6] == (0, 0, 0, 1)
    assert match.times_old.tolist() == []
    assert match.times_new.tolist() == [4.0]
    # the old dump has the time twice: its state after the change is compared
    match = matched_rows(np.array([0.0, 1, 2, 2, 3]), T_OLD)
    assert match.old.tolist() == [0, 1, 3, 4]
    assert match.new.tolist() == [0, 1, 2, 3]
    assert match[2:6] == (1, 0, 0, 0)
    # a change at the last time
    match = matched_rows(T_OLD, np.array([0.0, 1, 2, 3, 3]))
    assert match.old.tolist() == [0, 1, 2, 3]
    assert match.new.tolist() == [0, 1, 2, 4]
    assert match[2:6] == (0, 1, 0, 0)


def test_a_state_before_a_change_differs_although_the_times_differ(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = written(tmp_path, "o.pkl", timecourse([0, 1, 2, 2, 3], [1, 2, 3, 3.5, 4]))
    times = [0, 1, 2, 2, 3, 4]
    same = written(tmp_path, "s.pkl", timecourse(times, [1, 2, 3, 3.5, 4, 5]))
    other = written(tmp_path, "n.pkl", timecourse(times, [1, 2, 99, 3.5, 4, 5]))
    # the new output time is expected, the states at 2 are compared
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"results": ["E/t/time"]}}), encoding="utf-8"
    )
    args = ["--threshold", "1e-12", "--renames", str(renames)]
    assert verify(["compare", old, same, *args]) == 0
    assert verify(["compare", old, other, *args]) == 1


def test_a_new_output_time_is_a_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = written(tmp_path, "o.pkl", timecourse([0, 1, 2, 3], [1, 2, 3, 4]))
    new = written(tmp_path, "n.pkl", timecourse([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))
    capsys.readouterr()
    assert verify(["compare", old, new, "--threshold", "1e-12"]) == 1
    out = capsys.readouterr().out
    assert "1 rows of the new dump without a time in the old dump (0 expected)" in out
    assert "  times of E/t: 1 rows only in new, at 4\n" in out
    # the pattern of the variable time of the task names the new output times
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"results": ["E/t/time"]}}), encoding="utf-8"
    )
    assert verify(["compare", old, new, "--renames", str(renames)]) == 0
    out = capsys.readouterr().out
    assert "1 rows of the new dump without a time in the old dump (1 expected)" in out
    assert "  times of E/t: 1 rows only in new, at 4 (expected)" in out
    # a time only the old dump has is a difference, also with the pattern
    assert verify(["compare", new, old, "--renames", str(renames)]) == 1
    assert "  times of E/t: 1 rows only in old, at 4\n" in capsys.readouterr().out


def dosed(times: np.ndarray, rate: float = 0.5) -> np.ndarray:
    """A decay with a second dose at 2: a trajectory of an adaptive grid.

    `times` has the time of the dose twice, the state before and after it.
    """
    before = np.exp(-rate * times)
    after = (math.exp(-2 * rate) + 1.0) * np.exp(-rate * (times - 2.0))
    values = np.where(times < 2.0, before, after)
    values[int(np.flatnonzero(times == 2.0)[0])] = math.exp(-2 * rate)
    return values


#: two adaptive grids of one simulation: other steps, the dose at 2 in both;
#: dense enough that the spread of their interpolants is below 1e-6 (the
#: resolution which `compare` needs at its threshold)
GRID_A = np.concatenate([np.linspace(0, 2, 801), np.linspace(2, 10, 3201)])
GRID_B = np.concatenate([np.linspace(0, 2, 1001), np.linspace(2, 10, 4001)])


def test_two_grids_of_one_trajectory_do_not_differ(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row at a time only one dump has is compared with the other interpolated."""
    old = written(tmp_path, "o.pkl", timecourse(GRID_A, dosed(GRID_A)))
    new = written(tmp_path, "n.pkl", timecourse(GRID_B, dosed(GRID_B)))
    capsys.readouterr()
    assert verify(["compare", old, new]) == 0, capsys.readouterr().out
    assert verify(["compare", new, old]) == 0
    out = capsys.readouterr().out
    assert "0 rows of the old dump without a time in the new dump" in out
    assert "0 unresolved on this grid" in out
    match = matched_rows(GRID_B, GRID_A)
    assert match.interpolated_old.size > GRID_B.size / 2
    assert (
        f"rows at other times: {match.interpolated_old.size} rows of the old dump "
        f"and {match.interpolated_new.size} of the new dump compared with the "
        "other dump interpolated at their times" in out
    )


def test_a_trajectory_on_another_grid_differs(tmp_path: Path) -> None:
    """The interpolation leaves a difference of the trajectory: a slower decay."""
    old = written(tmp_path, "o.pkl", timecourse(GRID_A, dosed(GRID_A)))
    new = written(tmp_path, "n.pkl", timecourse(GRID_B, dosed(GRID_B, rate=0.501)))
    assert verify(["compare", old, new]) == 1
    assert verify(["compare", new, old]) == 1


def test_a_difference_between_the_common_times_is_found(tmp_path: Path) -> None:
    """A change of the trajectory of 1e-4 which vanishes at the common times."""
    old = written(tmp_path, "o.pkl", timecourse(GRID_A, dosed(GRID_A)))
    for eps, code in ((1e-4, 1), (1e-7, 0)):
        bump = np.where(
            GRID_B <= 2,
            1 + eps * np.sin(np.pi * GRID_B / 2),
            1 + eps * np.sin(np.pi * (GRID_B - 2) / 8),
        )
        new = written(tmp_path, "n.pkl", timecourse(GRID_B, dosed(GRID_B) * bump))
        assert verify(["compare", old, new]) == code


def tasks(values: dict[str, Sequence[float]]) -> Dump:
    """A dump of tasks of one experiment with the output times 0 to 3."""
    results: dict = {}
    for task, task_values in values.items():
        results["E", task, "time"] = (("time",), np.array([0.0, 1.0, 2.0, 3.0]))
        results["E", task, "[A]"] = (("time",), np.array(task_values))
    return synthetic(results)


def test_the_noise_of_a_control_is_no_difference(tmp_path: Path) -> None:
    """A variable which is zero in a task is compared at the scale of the experiment."""
    dose = [1.0, 0.5, 0.25, 0.125]
    old = written(tmp_path, "o.pkl", tasks({"dose": dose, "control": [0, 1e-24, 0, 0]}))
    noise = tasks({"dose": dose, "control": [0, 3e-24, -1e-25, 0]})
    assert verify(["compare", old, written(tmp_path, "n.pkl", noise)]) == 0
    # a small value which is not noise is compared at its own scale
    small = written(
        tmp_path, "s.pkl", tasks({"dose": dose, "control": [0, 1e-6, 0, 0]})
    )
    other = tasks({"dose": dose, "control": [0, 1.1e-6, 0, 0]})
    assert verify(["compare", small, written(tmp_path, "m.pkl", other)]) == 1


def bateman(times: np.ndarray, ke: float = 0.2) -> np.ndarray:
    """An oral dose: absorption 3/hr, elimination `ke`."""
    ka = 3.0
    return ka / (ka - ke) * (np.exp(-ke * times) - np.exp(-ka * times))


#: two coarse adaptive grids of an oral dose over 48 hr, 20 steps each
COARSE_A = np.concatenate([[0.0], np.geomspace(0.01, 48, 20)])
COARSE_B = np.concatenate([[0.0], np.geomspace(0.013, 48, 20)])


@pytest.mark.parametrize("ke", [0.2, 0.2002])
def test_an_array_which_the_grid_cannot_resolve_is_no_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ke: float
) -> None:
    """The interpolation of 20 steps resolves about 1e-2, not the threshold.

    An elimination faster by 1e-3 (a change of 4.7e-4 of the scale) lies
    within the range of the interpolants and is reported as unresolved, as is
    the same trajectory: compare passes no array it cannot resolve.
    """
    old = written(tmp_path, "o.pkl", timecourse(COARSE_A, bateman(COARSE_A)))
    new = written(tmp_path, "n.pkl", timecourse(COARSE_B, bateman(COARSE_B, ke)))
    capsys.readouterr()
    assert verify(["compare", old, new]) == 1
    out = capsys.readouterr().out
    assert "0 differ by more than 1e-06" in out
    assert "1 unresolved on this grid" in out
    (line,) = [x for x in out.splitlines() if "E/t/[A]" in x and "resolution" in x]
    resolution = float(line.split()[0])
    assert 1e-3 < resolution < 1e-1
    # at a threshold above the resolution the array is resolved
    threshold = str(2 * resolution) if ke == 0.2 else "1.0"
    assert verify(["compare", old, new, "--threshold", threshold]) == 0


def test_the_interpolation_does_not_cross_a_change() -> None:
    """A row is interpolated between the rows of the segment of its time."""
    old_times = np.array([0.0, 1.0, 2.0, 2.0, 3.0])
    new_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 4.0])
    match = matched_rows(old_times, new_times)
    assert match.old.tolist() == [0, 1, 2, 3, 4]
    assert match.interpolated_old.tolist() == []
    # the new rows between the common times are interpolated in the old dump,
    # the row after its last time is not
    assert match.interpolated_new.tolist() == [1, 3, 6]
    assert match.unmatched_new == 1
    assert match.times_new.tolist() == [4.0]
    # a repeated time which the other dump does not have is not interpolated
    match = matched_rows(np.array([0.0, 1.0, 1.5, 1.5, 3.0]), np.array([0.0, 1, 3]))
    assert match.interpolated_old.tolist() == []
    assert match.unmatched == 2
    # nor a row after the last time of the other dump
    match = matched_rows(np.array([0.0, 1.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0]))
    assert match.interpolated_old.tolist() == []
    assert match.interpolated_new.tolist() == [1]
    assert match.unmatched == 1


def test_close_times_are_not_matched() -> None:
    """Two rows of times 1e-7 apart are interpolated, not compared as one time."""
    match = matched_rows(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0 + 1e-7, 2.0]))
    assert match.old.tolist() == [0, 2]
    assert match.interpolated_old.tolist() == [1]
    assert match.interpolated_new.tolist() == [1]


def test_repeated_times_of_the_old_dump_are_left_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = written(tmp_path, "o.pkl", timecourse([0, 1, 2, 2, 3], [1, 2, 9, 3, 4]))
    new = written(tmp_path, "n.pkl", timecourse([0, 1, 2, 3], [1, 2, 3, 4]))
    capsys.readouterr()
    assert verify(["compare", old, new, "--threshold", "1e-12"]) == 0
    assert (
        "0 rows of the new dump and 1 of the old dump left out"
        in capsys.readouterr().out
    )


def test_the_state_after_a_change_at_the_last_time_is_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # a dump with the state after a change only, at the last output time
    old = written(tmp_path, "o.pkl", timecourse([0, 1, 2, 3], [1, 2, 3, 10]))
    times = [0, 1, 2, 3, 3]
    new = written(tmp_path, "n.pkl", timecourse(times, [1, 2, 3, 4, 10]))
    swapped = written(tmp_path, "w.pkl", timecourse(times, [1, 2, 3, 10, 4]))
    capsys.readouterr()
    assert verify(["compare", old, new, "--threshold", "1e-12"]) == 0
    assert "changes: E 1 rows" in capsys.readouterr().out
    assert verify(["compare", old, swapped, "--threshold", "1e-12"]) == 1


def test_a_scan_of_one_point_compares_with_a_simulation(tmp_path: Path) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    old = synthetic(
        {
            ("E", "task_x", "time"): (("_point", "dim_ref"), T_OLD[:, np.newaxis]),
            ("E", "task_x", "[A]"): (("_point", "dim_ref"), values[:, np.newaxis]),
        }
    )
    new = synthetic(
        {
            ("E", "task_x", "time"): (("time",), T_OLD),
            ("E", "task_x", "[A]"): (("time",), values),
        }
    )
    assert (
        verify(
            [
                "compare",
                written(tmp_path, "o.pkl", old),
                written(tmp_path, "n.pkl", new),
            ]
        )
        == 0
    )


def test_an_array_without_a_time_is_compared_relative_to_its_scale(
    tmp_path: Path,
) -> None:
    # the values at the end of the points of a scan, e.g. a clearance per renal
    # function, have a dimension of the scan and no time
    def dump(values: list[float]) -> Dump:
        return synthetic(
            {
                ("E", "task_x", "time"): (("time",), T_OLD),
                ("E", "task_x", "cl"): (("dim_renal",), np.array(values)),
            }
        )

    old = written(tmp_path, "o.pkl", dump([2.0, 4.0, 10.0]))
    close = written(tmp_path, "c.pkl", dump([2.0 + 1e-10, 4.0, 10.0]))
    far = written(tmp_path, "f.pkl", dump([2.1, 4.0, 10.0]))
    nan = written(tmp_path, "n.pkl", dump([np.nan, 4.0, 10.0]))
    assert verify(["compare", old, close]) == 0
    assert verify(["compare", old, far]) == 1
    assert verify(["compare", old, nan]) == 1


def test_arrays_without_a_time_of_other_dimensions_differ(tmp_path: Path) -> None:
    def dump(dim: str) -> Dump:
        return synthetic(
            {
                ("E", "task_x", "time"): (("time",), T_OLD),
                ("E", "task_x", "cl"): ((dim,), np.array([2.0, 4.0])),
            }
        )

    old = written(tmp_path, "o.pkl", dump("dim_renal"))
    assert verify(["compare", old, written(tmp_path, "n.pkl", dump("dim_q"))]) == 1


def test_arrays_of_other_dimensions_differ(tmp_path: Path) -> None:
    new = new_scan()
    _, values = new["results"]["E", "task_x", "[A]"]
    new["results"]["E", "task_x", "[A]"] = (("dim_q", "time"), values)
    old = written(tmp_path, "o.pkl", old_scan())
    assert verify(["compare", old, written(tmp_path, "n.pkl", new)]) == 1


def test_arrays_of_other_sizes_differ(tmp_path: Path) -> None:
    new = new_scan()
    _, values = new["results"]["E", "task_x", "[A]"]
    new["results"]["E", "task_x", "[A]"] = (
        ("dim_k", "time"),
        np.vstack([values, values]),
    )
    old = written(tmp_path, "o.pkl", old_scan())
    assert verify(["compare", old, written(tmp_path, "n.pkl", new)]) == 1


def test_the_ids_of_the_old_dump_become_sids(tmp_path: Path) -> None:
    def figure(sid: str, task: str) -> str:
        spec = {"sid": sid, "curves": [{"task_id": task, "name": "po 2.5"}]}
        return json.dumps(spec, sort_keys=True)

    def dump(task: str, dataset: str, fig: str, mapping: str) -> Dump:
        return synthetic(
            {
                ("E", task, "time"): (("time",), T_OLD),
                ("E", task, "[A]"): (("time",), T_OLD),
            },
            datasets={("E", dataset, "mean"): np.array([1.0])},
            figures={"E": {fig: figure(fig, task)}},
            ranges={"E": {mapping: (0.0, 1.0, 0.0, 1.0, "hr")}},
        )

    old = written(tmp_path, "o.pkl", dump("task_po 2.5", "Fig1_A b", "fig 1", "fm_a b"))
    new = written(tmp_path, "n.pkl", dump("task_po_2p5", "Fig1_A_b", "fig_1", "fm_a_b"))
    renames = tmp_path / "renames.json"
    renames.write_text(json.dumps({"sids": True}), encoding="utf-8")
    assert verify(["compare", old, new]) == 1
    assert verify(["compare", old, new, "--renames", str(renames)]) == 0


def test_the_ids_of_a_mapping_description_are_renamed() -> None:
    description = (
        "fm_a b|Fig1_A b|None|time|mean|4|mean_sd|None"
        "|None|task_po 2.5|time|A|None|None|None|weight=None|[('route', 'IV')]"
    )
    renamed = apply_renames(synthetic(mappings={"E": [description]}), {"sids": True})
    assert renamed["mappings"]["E"] == [
        "fm_a_b|Fig1_A_b|None|time|mean|4|mean_sd|None"
        "|None|task_po_2p5|time|A|None|None|None|weight=None|[('route', 'IV')]"
    ]


def test_a_key_with_a_bar_cannot_be_renamed() -> None:
    description = (
        "fm|a|x|Fig1|None|time|mean|4|mean_sd|None"
        "|None|task|time|A|None|None|None|weight=None|[]"
    )
    with pytest.raises(ValueError, match="cannot be renamed"):
        apply_renames(synthetic(mappings={"E": [description]}), {"sids": True})


def test_two_ids_with_one_new_id_are_an_error() -> None:
    dump = synthetic(
        datasets={
            ("E", "a b", "x"): np.array([1.0]),
            ("E", "a-b", "x"): np.array([2.0]),
        }
    )
    with pytest.raises(ValueError, match="same new id"):
        apply_renames(dump, {"sids": True})


@pytest.mark.parametrize(
    "parts",
    [
        {
            "results": {
                ("E", "task a b", "x"): (("time",), T_OLD),
                ("E", "task a-b", "x"): (("time",), T_OLD),
            }
        },
        {"figures": {"E": {"fig a": '{"sid": "fig a"}', "fig-a": '{"sid": "fig-a"}'}}},
        {"ranges": {"E": {"fm a": "no x values", "fm-a": "no x values"}}},
    ],
    ids=["results", "figures", "ranges"],
)
def test_two_ids_of_a_part_with_one_new_id_are_an_error(parts: dict) -> None:
    with pytest.raises(ValueError, match="same new id"):
        apply_renames(synthetic(**parts), {"sids": True})


def test_two_keys_with_one_key_after_the_strings_are_an_error() -> None:
    dump = synthetic(
        datasets={("E", "d", "x1"): np.array([1.0]), ("E", "d", "x2"): np.array([2.0])}
    )
    with pytest.raises(ValueError, match="same new key"):
        apply_renames(dump, {"strings": {"x1": "x2"}})


def test_the_strings_do_not_rename_the_parts_of_a_dump() -> None:
    dump = synthetic(datasets={("E", "a", "x"): np.array([1.0])})
    renamed = apply_renames(dump, {"strings": {"a": "b"}})
    assert renamed.keys() == dump.keys()
    assert list(renamed["datasets"]) == [("E", "b", "x")]


def test_the_renames_do_not_change_the_old_dump() -> None:
    descriptions = ["b", "a"]
    dump = synthetic(mappings={"E": descriptions})
    before = copy.deepcopy(dump)
    result = apply_renames(dump, {})
    assert result["mappings"]["E"] == ["a", "b"]
    assert dump["mappings"]["E"] is descriptions
    assert descriptions == ["b", "a"]
    assert dump.keys() == before.keys()


def test_an_expected_difference_is_not_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = written(
        tmp_path, "o.pkl", synthetic(datasets={("E", "d", "mean"): np.array([1.0])})
    )
    new = written(
        tmp_path, "n.pkl", synthetic(datasets={("E", "d", "mean"): np.array([2.0])})
    )
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"datasets": ["E/d/*"]}}), encoding="utf-8"
    )
    capsys.readouterr()
    assert verify(["compare", old, new, "--renames", str(renames)]) == 0
    assert "datasets: 1 old, 1 new, 1 differ (1 expected)" in capsys.readouterr().out


def mapping(key: str, y: str, task: str) -> str:
    """The description of a fit mapping of the column y to [A] of a task."""
    return (
        f"{key}|d|None|time|{y}|4|None|None|None|{task}|time|[A]|None|None|None"
        "|weight=None|[]"
    )


def test_an_expected_mapping_difference_is_one_mapping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = synthetic(
        mappings={"E": [mapping("fm1", "A", "t"), mapping("fm2", "A", "t")]}
    )
    new = synthetic(
        mappings={"E": [mapping("fm1", "B", "t"), mapping("fm2", "A", "u")]}
    )
    a, b = written(tmp_path, "o.pkl", old), written(tmp_path, "n.pkl", new)
    renames = tmp_path / "renames.json"
    capsys.readouterr()
    for patterns, code in ((["E"], 1), (["E/fm1"], 1), (["E/fm*"], 0)):
        renames.write_text(
            json.dumps({"expected": {"mappings": patterns}}), encoding="utf-8"
        )
        assert verify(["compare", a, b, "--renames", str(renames)]) == code, patterns
    out = capsys.readouterr().out
    assert "mappings: 2 old, 2 new, 2 differ (1 expected)" in out
    assert "expected difference which does not occur: mappings: E" in out
    assert (
        f"  E/fm2\n    - {mapping('fm2', 'A', 't')}\n    + {mapping('fm2', 'A', 'u')}"
        in out
    )


def test_two_mappings_of_one_key_are_an_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dump = synthetic(mappings={"E": [mapping("fm", "A", "t"), mapping("fm", "B", "t")]})
    path = written(tmp_path, "d.pkl", dump)
    assert verify(["compare", path, path]) == 2
    assert "Two fit mappings of E have the key fm" in caplog.text


def test_a_bracket_of_a_pattern_is_literal(tmp_path: Path) -> None:
    def dump(a: float, bracket: float) -> Dump:
        return synthetic(
            {
                ("E", "t", "time"): (("time",), T_OLD),
                ("E", "t", "A"): (("time",), np.full(4, a)),
                ("E", "t", "[A]"): (("time",), np.full(4, bracket)),
            }
        )

    old = written(tmp_path, "o.pkl", dump(1.0, 1.0))
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"results": ["E/t/[A]"]}}), encoding="utf-8"
    )
    other_bracket = written(tmp_path, "b.pkl", dump(1.0, 2.0))
    other_plain = written(tmp_path, "p.pkl", dump(2.0, 1.0))
    assert verify(["compare", old, other_bracket, "--renames", str(renames)]) == 0
    assert verify(["compare", old, other_plain, "--renames", str(renames)]) == 1


def test_an_expected_difference_which_does_not_occur_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = written(
        tmp_path, "s.pkl", synthetic(datasets={("E", "d", "mean"): np.array([1.0])})
    )
    renames = tmp_path / "renames.json"
    renames.write_text(
        json.dumps({"expected": {"figures": ["E/fig_x"]}}), encoding="utf-8"
    )
    capsys.readouterr()
    assert verify(["compare", same, same, "--renames", str(renames)]) == 1
    assert (
        "expected difference which does not occur: figures: E/fig_x"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "renames",
    [
        {"ids": {"a": 1}},
        {"sids": "yes"},
        {"figure_fields": "width"},
        {"expected": {"results": "E/*"}},
        {"expected": {"plots": ["E"]}},
    ],
)
def test_a_renames_file_with_wrong_new_parts_is_an_error(
    tmp_path: Path, renames: dict
) -> None:
    path = tmp_path / "renames.json"
    path.write_text(json.dumps(renames), encoding="utf-8")
    with pytest.raises(ValueError):
        load_renames(str(path))


def test_a_figure_of_one_dump_only_is_a_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    both = {"f": json.dumps({"sid": "f"}), "g": json.dumps({"sid": "g"})}
    one = {"f": both["f"]}
    old = written(tmp_path, "o.pkl", synthetic(figures={"E": both}))
    new = written(tmp_path, "n.pkl", synthetic(figures={"E": one}))
    assert verify(["compare", old, new]) == 1
    assert "figures: 2 old, 1 new, 1 differ (0 expected)" in capsys.readouterr().out
    assert verify(["compare", new, old]) == 1


def test_png_pairs_the_figures_of_renamed_keys(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    write_png(old / "A" / "A_fig 2.5.png")
    write_png(new / "A" / "A_fig_2p5.png")
    assert verify(["png", str(old), str(new)]) == 1
    assert verify(["png", "--sids", str(old), str(new)]) == 0
