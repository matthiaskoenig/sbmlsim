"""The cross-engine benchmark, `scripts/engine_benchmark.py`.

These tests prove the shape of the measurement and of its tables, not the
speed of the engines: they measure the smallest benchmark there is (two
repetitions, one round, a scan of four points) and assert what the tables
say of it. The numbers of the documentation are measured by hand with the
command line of the script, see its module docstring.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import scripts.engine_benchmark as engine_benchmark
from sbmlsim.engine import Engine, UnsupportedModelError
from sbmlsim.model import SBMLModel
from sbmlsim.result.timecourse import TimecourseResult
from scripts.engine_benchmark import (
    MIN_ROUNDS,
    MODELS,
    BenchmarkResult,
    BenchmarkRow,
    Measurement,
    ModelConfiguration,
    agreement,
    main,
    markdown,
    measure,
)
from tests.engines import require


@pytest.fixture(scope="module")
def measured() -> BenchmarkResult:
    """Measure the repressilator with every engine which is installed."""
    return measure(
        ["repressilator"],
        list(Engine),
        repetitions=2,
        rounds=1,
        warmups=1,
        points=4,
        workers=1,
    )


def _tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Read every markdown table of a text by the heading which introduces it.

    Args:
        text: the rendered tables and their legend.

    Returns:
        The rows of every table as the cells of their columns, by the text of
        the `###` heading above it.
    """
    tables: dict[str, list[dict[str, str]]] = {}
    heading, header = "", []
    for line in text.splitlines():
        if line.startswith("### "):
            heading, header = line[4:], []
        elif line.startswith("|") and not header:
            header = [cell.strip() for cell in line.split("|")[1:-1]]
            tables[heading] = []
        elif line.startswith("|") and not line.startswith("|---"):
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            tables[heading].append(dict(zip(header, cells, strict=True)))
        elif not line.strip():
            header = []
    return tables


def _table(text: str, heading: str) -> list[dict[str, str]]:
    """Read the rows of one table of a markdown.

    Args:
        text: the rendered tables and their legend.
        heading: the text of the `###` heading above the table.

    Returns:
        The rows as the cells of their columns.
    """
    return _tables(text)[heading]


def _cells(text: str) -> list[dict[str, str]]:
    """Read the tables of a markdown as one row per model and engine.

    The tables are narrow and share their rows, so a test reads a cell of
    any of them by its column, see `markdown`.

    Args:
        text: the rendered tables and their legend.

    Returns:
        One mapping per model and engine, the columns of every table it has
        a row in, in the order of the rows.
    """
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for heading, rows in _tables(text).items():
        if heading == "What every model was measured with":
            continue
        for cells in rows:
            key = (cells["model"], cells["engine"])
            merged.setdefault(key, {}).update(cells)
    return list(merged.values())


def _measurement(mean: float, sd: float | None = None) -> Measurement:
    """Get a measurement of a handmade result.

    Args:
        mean: the mean of its rounds.
        sd: their standard deviation, `None` for a cell of one round.

    Returns:
        The measurement.
    """
    return Measurement(mean=mean, sd=sd, rounds=1 if sd is None else MIN_ROUNDS)


def _handmade() -> BenchmarkResult:
    """Get a result of a parallel run of two engines, measured by nothing."""
    return BenchmarkResult(
        rows={
            ("hctz", Engine.CORE): BenchmarkRow(
                load_s=_measurement(0.0122),
                compile_s=_measurement(0.003),
                instantiate_s=_measurement(0.004),
                first_compile_s=0.122,
                simulate_s=_measurement(0.00289),
                scan_serial_s=_measurement(1.137),
                scan_parallel_s=_measurement(0.258, 0.0121),
                max_relative_difference=2.4e-11,
            ),
            ("hctz", Engine.ROADRUNNER): BenchmarkRow(
                load_s=_measurement(0.215),
                first_compile_s=0.264,
                simulate_s=_measurement(0.00357),
                scan_serial_s=_measurement(1.643),
                scan_parallel_s=_measurement(1.583),
                max_relative_difference=0.0,
                max_absolute_difference=0.0,
            ),
        },
        repetitions=20,
        rounds=3,
        warmups=3,
        points=1000,
        workers=8,
        configurations={
            "hctz": ModelConfiguration(
                output_times=101, points=1000, repetitions=20, warmups=3, rounds=3
            )
        },
        skipped={Engine.AMICI: "The engine 'amici' needs AMICI"},
        versions={"python": "3.14.1", "sbmlsim": "0.14.0"},
    )


def test_the_benchmark_measures_every_engine_it_finds() -> None:
    """Every phase of a measured cell has its number: the load, a simulation and a scan."""
    result = measure(
        ["repressilator"],
        [Engine.ROADRUNNER],
        repetitions=2,
        rounds=1,
        warmups=1,
        points=4,
        workers=1,
    )
    row = result.rows[("repressilator", Engine.ROADRUNNER)]
    assert row.load_s is not None and row.load_s.mean > 0
    assert row.simulate_s is not None and row.simulate_s.mean > 0
    assert row.scan_serial_s is not None and row.scan_serial_s.mean > 0
    assert row.first_compile_s is not None and row.first_compile_s > 0
    assert row.note is None


def test_every_table_has_a_row_per_model_and_engine(measured: BenchmarkResult) -> None:
    """Every table of the markdown has the same rows in the same order.

    The tables are narrow so that a page can display them, see `markdown`,
    and a reader follows a row down the tables, so they share their rows.
    """
    tables = _tables(markdown(measured))
    expected = [(name, engine.value) for name, engine in measured.rows]
    assert [row["model"] for row in tables["What every model was measured with"]] == [
        "repressilator"
    ]
    for heading in ("Loading a model", "One simulation and a scan"):
        assert [(row["model"], row["engine"]) for row in tables[heading]] == expected
    assert [(row["model"], row["engine"]) for row in _cells(markdown(measured))] == (
        expected
    )


def test_an_engine_which_is_not_installed_is_skipped() -> None:
    """An engine which is not installed has no row, its reason is recorded instead.

    AMICI is built from source and is not installed in most environments;
    where it is installed it is measured, which `require` of `tests.engines`
    decides.
    """
    result = measure(
        ["repressilator"],
        [Engine.ROADRUNNER, Engine.AMICI],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
    )
    assert ("repressilator", Engine.ROADRUNNER) in result.rows
    if Engine.AMICI in result.skipped:
        assert ("repressilator", Engine.AMICI) not in result.rows
        assert "sbmlsim[amici]" in result.skipped[Engine.AMICI]
        return
    require(Engine.AMICI)
    assert ("repressilator", Engine.AMICI) in result.rows


def test_the_agreement_with_roadrunner_is_reported() -> None:
    """The engine core agrees with roadrunner on the model it is measured on."""
    require(Engine.CORE)
    result = measure(
        ["repressilator"],
        [Engine.CORE],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
    )
    row = result.rows[("repressilator", Engine.CORE)]
    assert row.max_relative_difference is not None
    assert row.max_relative_difference < 1e-6
    # the reference of the agreement was simulated, so its version is named
    # although roadrunner was not one of the measured engines
    assert "libroadrunner" in result.versions
    assert "libroadrunner" in markdown(result)


def test_a_column_which_is_zero_in_roadrunner_is_compared_absolutely() -> None:
    """A column which is zero in roadrunner is compared absolutely, never relatively.

    A relative difference of such a column is undefined, so a naive
    `|c-u|/|c|` is infinite or not a number there; the rule of the benchmark
    reports it in the absolute column instead.
    """
    columns = ("time", "x", "zero")
    reference = TimecourseResult(
        columns=columns, values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    )
    other = TimecourseResult(
        columns=columns, values=np.array([[0.0, 1.0, 1e-18], [1.0, 2.0000002, 0.0]])
    )
    relative, absolute = agreement(other, reference)
    assert relative is not None and absolute is not None
    # the largest difference of a column relative to the largest magnitude of
    # that column, i.e. 2e-7 of the 2.0 of the column `x`
    assert relative == pytest.approx(1e-7, rel=1e-9)
    assert absolute == 1e-18
    # the naive relative difference per value, which the rule replaces
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = np.abs(other.values - reference.values) / np.abs(reference.values)
    assert not np.isfinite(np.nanmax(naive))


def test_a_column_which_is_zero_in_both_is_no_difference() -> None:
    """A column which is zero in both engines is a zero absolute difference."""
    columns = ("time", "zero")
    values = np.array([[0.0, 0.0], [1.0, 0.0]])
    relative, absolute = agreement(
        TimecourseResult(columns=columns, values=values.copy()),
        TimecourseResult(columns=columns, values=values),
    )
    assert relative == 0.0
    assert absolute == 0.0


def test_the_agreement_of_a_measured_engine_is_a_number(
    measured: BenchmarkResult,
) -> None:
    """No cell of the agreement is infinite or not a number, in the table either."""
    for key, row in measured.rows.items():
        for difference in (row.max_relative_difference, row.max_absolute_difference):
            assert difference is None or math.isfinite(difference), key
    for row_cells in _cells(markdown(measured)):
        for column in ("max rel. diff.", "max abs. diff."):
            cell = row_cells[column]
            assert cell == "" or math.isfinite(float(cell)), (row_cells, column)


def test_a_parallel_scan_is_measured_on_the_workers_of_the_run() -> None:
    """Two workers add the parallel scan to the table, which runs it in a pool."""
    result = measure(
        ["repressilator"],
        [Engine.ROADRUNNER],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=2,
    )
    row = result.rows[("repressilator", Engine.ROADRUNNER)]
    assert row.scan_parallel_s is not None and row.scan_parallel_s.mean > 0
    assert "2 python processes" in markdown(result)


def test_the_legend_says_whether_a_parallel_scan_ran_on_processes_or_threads() -> None:
    """A parallel scan of the engine core runs on threads, every other one on processes.

    It is the same for every row of a run, so it is one line of the legend
    and not a column which repeats it; the tables are narrow.
    """
    text = markdown(_handmade())
    rows = {cells["engine"]: cells for cells in _cells(text)}
    assert rows["roadrunner"]["scan parallel s"] == "1.58"
    assert "parallel on" not in rows["roadrunner"]
    assert "8 python processes for roadrunner" in text
    assert "threads of the engine" in text
    assert "do not scale alike" in text


def test_a_measurement_which_was_not_made_is_an_empty_cell() -> None:
    """A column an engine has no measurement of is an empty cell of its row."""
    rows = {cells["engine"]: cells for cells in _cells(markdown(_handmade()))}
    # the compilation of the model is part of the load of roadrunner
    assert rows["roadrunner"]["compile s"] == ""
    assert rows["roadrunner"]["instantiate s"] == ""
    assert rows["roadrunner"]["load s"] != ""
    # the row of the engine core has no absolute difference, i.e. roadrunner
    # had no column which is identically zero
    assert rows["core"]["max abs. diff."] == ""
    assert rows["core"]["compile s"] != ""


def test_the_simulation_of_a_row_is_rendered_in_milliseconds() -> None:
    """`simulate ms` is the seconds of the row in milliseconds, the one column which converts."""
    rows = {cells["engine"]: cells for cells in _cells(markdown(_handmade()))}
    # the rows of `_handmade`, 0.00357 s and 0.00289 s
    assert rows["roadrunner"]["simulate ms"] == "3.57"
    assert rows["core"]["simulate ms"] == "2.89"
    # the columns of seconds are seconds
    assert rows["roadrunner"]["load s"] == "0.215"
    assert rows["core"]["instantiate s"] == "0.004"


def test_a_model_an_engine_refuses_keeps_its_row_with_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model an engine refuses is a row of its refusal and not a row which is missing."""

    def refuse(*args: object, **kwargs: object) -> SBMLModel:
        raise UnsupportedModelError("The engine 'core' cannot compile the model 'demo'")

    monkeypatch.setattr(engine_benchmark, "load_model", refuse)
    result = measure(
        ["demo"],
        [Engine.ROADRUNNER],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
    )
    row = result.rows[("demo", Engine.ROADRUNNER)]
    assert row.note is not None and "cannot compile" in row.note
    assert row.load_s is None
    assert row.simulate_s is None
    assert row.scan_serial_s is None
    cells = _cells(markdown(result))[0]
    assert "cannot compile" in cells["why"]
    assert cells["load s"] == ""
    assert cells["simulate ms"] == ""
    assert cells["max rel. diff."] == ""


@pytest.mark.parametrize(
    ("function", "error"),
    [
        ("_tight", UnsupportedModelError("The engine 'core' refuses the plan")),
        ("agreement", ValueError("The column 'S1' has 3 rows and 4 rows")),
    ],
)
def test_a_comparison_which_fails_keeps_its_row_with_the_problem(
    function: str, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan an engine refuses and an output of other rows are noted, not raised.

    Both are the measurement of one cell: its row keeps what its load
    measured and the problem as its note, and the cell is not timed, so one
    awkward engine does not end the run of every model.
    """

    def refuse(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(engine_benchmark, function, refuse)
    result = measure(
        ["demo"],
        [Engine.ROADRUNNER],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
    )
    row = result.rows[("demo", Engine.ROADRUNNER)]
    assert row.note == str(error)
    # the load is measured before the comparison and is kept
    assert row.load_s is not None
    assert row.simulate_s is None
    assert row.scan_serial_s is None
    assert row.max_relative_difference is None
    cells = _cells(markdown(result))[0]
    assert cells["why"] == str(error)
    assert cells["load s"] != ""
    assert cells["simulate ms"] == ""


def test_an_engine_which_is_not_installed_is_named_below_the_table() -> None:
    """The table says which engines were not measured, so no row is read as a failure."""
    text = markdown(_handmade())
    assert "amici" in text
    assert "needs AMICI" in text


def test_the_command_line_writes_the_table_and_the_json(tmp_path: Path) -> None:
    """`--out` writes the table and `--json` the measurement it was rendered from."""
    table, payload = tmp_path / "table.md", tmp_path / "benchmark.json"
    code = main(
        [
            "--models",
            "repressilator",
            "--engines",
            "roadrunner",
            "--repetitions",
            "1",
            "--rounds",
            "1",
            "--points",
            "2",
            "--workers",
            "1",
            "--out",
            str(table),
            "--json",
            str(payload),
        ]
    )
    assert code == 0
    text = table.read_text(encoding="utf-8")
    assert text.startswith("### What every model was measured with")
    assert "| model | engine | load s |" in text
    written = json.loads(payload.read_text(encoding="utf-8"))
    assert written["points"] == 2
    assert [(row["model"], row["engine"]) for row in written["rows"]] == [
        ("repressilator", "roadrunner")
    ]
    assert written["rows"][0]["load_s"]["mean"] > 0
    assert written["configurations"]["repressilator"]["points"] == 2


def test_every_model_of_the_benchmark_is_a_file_which_is_scanned() -> None:
    """The models are the four packaged ones and the large model of `benchmarks/`."""
    assert list(MODELS) == [
        "repressilator",
        "demo",
        "midazolam",
        "hctz",
        "perfusion",
    ]
    for name, benchmark in MODELS.items():
        assert benchmark.source.exists(), name
        # the scanned target is an entity of the model
        sbml = benchmark.source.read_text(encoding="utf-8")
        assert f'id="{benchmark.target}"' in sbml, name
        assert benchmark.low < benchmark.high, name


def test_a_model_which_is_no_model_of_the_benchmark_names_the_models() -> None:
    """A name which is no model of the benchmark is refused with the known names."""
    with pytest.raises(ValueError, match="repressilator"):
        measure(
            ["no_such_model"],
            [Engine.ROADRUNNER],
            repetitions=1,
            rounds=1,
            warmups=0,
            points=2,
            workers=1,
        )


def test_cold_reports_the_first_run_of_every_scan() -> None:
    """`--cold` times the run before the timed ones and gives it columns of its own."""
    result = measure(
        ["repressilator"],
        [Engine.ROADRUNNER],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=2,
        cold=True,
    )
    row = result.rows[("repressilator", Engine.ROADRUNNER)]
    assert row.scan_serial_cold_s is not None and row.scan_serial_cold_s > 0
    assert row.scan_parallel_cold_s is not None and row.scan_parallel_cold_s > 0
    # the cold run of a parallel scan starts the pool and loads the model in
    # every worker, which the warm median does not pay
    assert row.scan_parallel_s is not None
    assert row.scan_parallel_cold_s > row.scan_parallel_s.mean
    text = markdown(result)
    cells = _cells(text)[0]
    assert cells["scan serial cold s"] != ""
    assert cells["scan parallel cold s"] != ""
    # the legend says that only the first cell of a run is a cold measurement
    assert "a run of its own" in text


def test_without_cold_the_table_has_no_cold_columns(measured: BenchmarkResult) -> None:
    """The cold columns are rendered for `--cold` only, the default table is unchanged."""
    header = list(_table(markdown(measured), "One simulation and a scan")[0])
    assert "scan serial cold s" not in header
    assert "scan parallel cold s" not in header
    assert all(
        row.scan_serial_cold_s is None and row.scan_parallel_cold_s is None
        for row in measured.rows.values()
    )


def test_steps_scans_every_model_on_that_output_grid() -> None:
    """`--steps` replaces the output of every model, which the legend names."""
    result = measure(
        ["repressilator"],
        [Engine.ROADRUNNER],
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
        steps=5,
    )
    assert result.steps == 5
    assert "`repressilator` to 100 in 6 output times" in markdown(result)
    # the model of the benchmark is unchanged, `with_steps` gives a copy
    assert MODELS["repressilator"].output_times == 101


def test_an_output_grid_keeps_the_rest_of_the_simulation_of_a_model() -> None:
    """`with_steps` is the output grid alone: the horizon and the dose are kept."""
    hctz = MODELS["hctz"]
    narrow = hctz.with_steps(9)
    assert narrow.output_times == 10
    assert narrow.simulation.end == hctz.simulation.end
    assert narrow.simulation.preinit_changes == hctz.simulation.preinit_changes
    assert (narrow.target, narrow.low, narrow.high) == (
        hctz.target,
        hctz.low,
        hctz.high,
    )


def test_an_output_grid_keeps_an_uncertain_pre_initialization() -> None:
    """`with_steps` rebuilds the simulation and must not drop its uncertain values.

    No packaged model of the benchmark has one, so nothing else would notice
    that the copy lost it.
    """
    from dataclasses import replace

    from sbmlsim.simulation import Simulation
    from sbmlsim.uncertainty import Uncertain

    hctz = MODELS["hctz"]
    dose = Uncertain(25.0, cv=0.2)
    model = replace(
        hctz,
        simulation=Simulation(
            end=hctz.simulation.end,
            steps=hctz.simulation.steps,
            preinit_changes={"PODOSE_hctz": dose},
        ),
    )
    assert model.simulation.uncertain_preinit == {"PODOSE_hctz": dose}
    narrow = model.with_steps(9)
    assert narrow.output_times == 10
    assert narrow.simulation.uncertain_preinit == {"PODOSE_hctz": dose}
    # the nominal value is the one a simulation simulates, in both
    assert narrow.simulation.preinit_changes == model.simulation.preinit_changes


def test_an_output_grid_which_is_not_positive_is_refused() -> None:
    """A grid of no steps is refused, as every other setting of the measurement is."""
    with pytest.raises(ValueError, match="steps"):
        measure(
            ["repressilator"],
            [Engine.ROADRUNNER],
            repetitions=1,
            rounds=1,
            warmups=0,
            points=2,
            workers=1,
            steps=0,
        )


def test_the_command_line_takes_the_cold_runs_and_the_output_grid(
    tmp_path: Path,
) -> None:
    """`--cold` and `--steps` reach the measurement and its JSON."""
    payload = tmp_path / "benchmark.json"
    code = main(
        [
            "--models",
            "repressilator",
            "--engines",
            "roadrunner",
            "--repetitions",
            "1",
            "--rounds",
            "1",
            "--points",
            "2",
            "--workers",
            "1",
            "--cold",
            "--steps",
            "5",
            "--json",
            str(payload),
        ]
    )
    assert code == 0
    written = json.loads(payload.read_text(encoding="utf-8"))
    assert written["cold"] is True
    assert written["steps"] == 5
    assert written["rows"][0]["scan_serial_cold_s"] > 0


def test_a_measurement_is_the_mean_and_the_deviation_of_its_rounds() -> None:
    """A cell is the mean of its rounds and their sample standard deviation."""
    measurement = Measurement.of([1.0, 2.0, 3.0])
    assert measurement.mean == 2.0
    assert measurement.sd == pytest.approx(1.0)
    assert measurement.rounds == 3
    # the unit of a cell is the renderer's, which scales both
    scaled = measurement.scaled(1e3)
    assert (scaled.mean, scaled.sd, scaled.rounds) == (2000.0, pytest.approx(1000.0), 3)


def test_a_measurement_of_too_few_rounds_has_no_deviation() -> None:
    """Below `MIN_ROUNDS` rounds a cell is its mean alone: a deviation would say nothing."""
    assert MIN_ROUNDS == 3
    for values in ([1.0], [1.0, 2.0]):
        measurement = Measurement.of(values)
        assert measurement.sd is None
        assert measurement.rounds == len(values)
    with pytest.raises(ValueError, match="at least one round"):
        Measurement.of([])


def test_a_cell_is_rendered_as_its_mean_and_its_deviation() -> None:
    """A cell with a deviation is `mean ± sd`, one without is the mean alone."""
    rows = {cells["engine"]: cells for cells in _cells(markdown(_handmade()))}
    # the parallel scan of the engine core, 0.258 s ± 0.0121 s
    assert rows["core"]["scan parallel s"] == "0.258 ± 0.012"
    # every other cell of the handmade result is one round
    assert rows["core"]["scan serial s"] == "1.14"
    assert rows["roadrunner"]["load s"] == "0.215"


def test_the_tables_say_what_every_model_was_measured_with() -> None:
    """The first table is the configuration of every model, which its cells are read with."""
    rows = _table(markdown(_handmade()), "What every model was measured with")
    assert rows == [
        {
            "model": "hctz",
            "output times": "101",
            "scan points": "1000",
            "simulations": "20",
            "warm ups": "3",
            "rounds": "3",
        }
    ]


def test_a_model_carries_its_own_points_and_rounds() -> None:
    """A large model is scanned over fewer points and measured in shorter rounds.

    One simulation of the perfusion model is tens of seconds in some
    engines, so the defaults of a run would measure it for hours.
    """
    perfusion = MODELS["perfusion"]
    assert perfusion.points == 8
    assert perfusion.rounds == MIN_ROUNDS
    assert perfusion.repetitions == 3
    assert perfusion.warmups == 1
    # a packaged model takes the configuration of the run
    for name in ("repressilator", "demo", "midazolam", "hctz"):
        model = MODELS[name]
        assert (model.points, model.repetitions, model.warmups, model.rounds) == (
            None,
            None,
            None,
            None,
        )


def test_the_configuration_of_a_model_wins_over_the_one_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model with its own points and rounds is measured with them, not with the run's."""
    from dataclasses import replace

    monkeypatch.setitem(
        MODELS, "demo", replace(MODELS["demo"], points=3, repetitions=1, rounds=1)
    )
    result = measure(
        ["demo"],
        [Engine.ROADRUNNER],
        repetitions=7,
        rounds=5,
        warmups=0,
        points=99,
        workers=1,
    )
    configuration = result.configurations["demo"]
    assert (configuration.points, configuration.repetitions, configuration.rounds) == (
        3,
        1,
        1,
    )
    # the run keeps its own settings, which every other model is measured with
    assert (result.points, result.repetitions, result.rounds) == (99, 7, 5)
    row = result.rows[("demo", Engine.ROADRUNNER)]
    assert row.load_s is not None and row.load_s.rounds == 1
    rows = _table(markdown(result), "What every model was measured with")
    assert rows[0]["scan points"] == "3"


def test_the_engines_of_the_table_begin_with_the_engine_core() -> None:
    """The order of the tables and of the figure is the order of the documentation."""
    assert engine_benchmark.ENGINES == (
        Engine.CORE,
        Engine.ROADRUNNER,
        Engine.COPASI,
        Engine.AMICI,
    )
    # every engine is measured, each of them once
    assert sorted(engine_benchmark.ENGINES, key=list(Engine).index) == list(Engine)


def test_the_command_line_draws_the_figure(tmp_path: Path) -> None:
    """`--figure` writes the comparison of the engines, which is a PNG file."""
    drawn = tmp_path / "figures" / "engines.png"
    code = main(
        [
            "--models",
            "repressilator",
            "--engines",
            "roadrunner",
            "--repetitions",
            "1",
            "--rounds",
            "1",
            "--points",
            "2",
            "--workers",
            "1",
            "--figure",
            str(drawn),
        ]
    )
    assert code == 0
    # the directory of the figure is created and the file is a PNG
    assert drawn.read_bytes().startswith(b"\x89PNG\r\n")


def test_a_figure_of_nothing_which_was_measured_is_refused(tmp_path: Path) -> None:
    """A measurement without a measured cell has nothing to draw and says so."""
    from scripts.engine_benchmark import figure

    empty = BenchmarkResult(
        rows={("hctz", Engine.CORE): BenchmarkRow(note="refused")},
        repetitions=1,
        rounds=1,
        warmups=0,
        points=2,
        workers=1,
    )
    with pytest.raises(ValueError, match="no cell to draw"):
        figure(empty, tmp_path / "engines.png")
