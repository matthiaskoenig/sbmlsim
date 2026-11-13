"""The study data behind the curves of a panel.

The model, the panel and the application of the tests are the fixtures of
`tests/serve/conftest.py`: the panel declares the cumulative amount of
hydrochlorothiazide in the urine of Beermann 1976 under its curve of the
urine, measured in mg and delivered in the `umol` of that curve.

The times of a series are delivered in the unit the x of a simulation of the
same panel is delivered in, which is `X_UNIT`, the time unit of the HCTZ
model, for a panel which declares no `x_unit` of its own.

A panel of another declaration is served by an application of its own
(`served`), and the series themselves are read with `panel_data`, which takes
the units of the panel from the runner (`series_of`): the runner of the module
resolves them on one warm model, as the application does.
"""

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from examples.panels.hctz import MR_HCTZ, URINE
from sbmlsim.panel import Panel, PanelCurve, PanelData, PanelSimulation
from sbmlsim.project import Project
from sbmlsim.serve.app import FAILED, create_app
from sbmlsim.serve.data import PanelSeries, panel_data
from sbmlsim.serve.payload import DATA_CONTRACT
from sbmlsim.serve.runner import (
    UNITLESS,
    ModelEntry,
    ModelRegistry,
    PanelRunner,
    PanelUnits,
)
from tests.serve.conftest import DATA, SIMULATE

#: the time unit of the HCTZ model, the one the x of its panels is in
X_UNIT = "min"
#: the times of the series of the panel, in the `hr` of its dataset
TIMES = [0.0, 12.0, 96.0, 168.0]
#: those times in the unit of the x of the panel
MINUTES = [time * 60.0 for time in TIMES]
#: the masses of the series of the panel, in the `mg` of its dataset
MASSES = [0.0, 3.01, 4.13, 4.135]
#: those masses as the substance of the model, in the `umol` of its curve
AMOUNTS = [mass / 297.7 * 1e3 for mass in MASSES]

#: a series of the group data of Patel 1984: the mean plasma concentration of
#: twelve subjects with its standard deviation, measured in `ng/ml` and
#: delivered as the substance of a volume
PATEL = PanelData(
    id="patel1984",
    label="Patel 1984 (25 mg, tablet)",
    dataset="Patel1984_Fig1",
    x="time",
    y="mean",
    yerr="mean_sd",
    sel={"label": "hctz25_tab"},
    conversion=MR_HCTZ,
    unit="uM",
)

#: the series of the crafted dataset of `crafted`, under the curve of the urine
MINE = PanelData(
    id="mine1999",
    label="Mine 1999",
    curve="Aurine_hctz",
    dataset="Mine1999_Tab1",
    x="time",
    y="mean",
    conversion=MR_HCTZ,
)
#: the same series of the crafted dataset whose times carry no unit
PLAIN = MINE.model_copy(update={"dataset": "Mine1999_Tab2"})
#: the times of both crafted datasets, in hours
MINE_TIMES = [12.0, 96.0]
#: those times in the unit of the x of the panel
MINE_MINUTES = [time * 60.0 for time in MINE_TIMES]
#: the values of the column of `Mine1999_Tab1` which carries no unit
RATIOS = [0.21, 0.29]

#: a model which declares no units at all, whose species therefore carry none
#: while its time is the `second` a model without time units falls back to
#: (`sbmlsim.units.UnitsInformation.from_sbml`)
PLAIN_MODEL = Path(__file__).parent.parent / "data" / "models" / "lotka_volterra.xml"
#: a panel of that model: a curve of a species without a unit and, under it,
#: the series of the crafted column which carries none either
PLAIN_PANEL = Panel(
    id="renal",
    title="Lotka Volterra",
    curves=(PanelCurve(id="prey", label="prey"),),
    data=(
        PanelData(
            id="mine1999",
            label="Mine 1999",
            curve="prey",
            dataset="Mine1999_Tab1",
            x="time",
            y="ratio",
        ),
    ),
    simulation=PanelSimulation(end=10.0, points=5),
)


@pytest.fixture(scope="module")
def runner(registry: ModelRegistry) -> PanelRunner:
    """One runner of the module, which resolves the units on a warm model."""
    return PanelRunner(registry)


@pytest.fixture
def crafted(project: Project, tmp_path: Path) -> Project:
    """A project whose data is the crafted datasets no study of the example has.

    `Mine1999_Tab1` carries a relative `cv` and a column without a unit,
    `Mine1999_Tab2` times without a unit, `Mine1999_Tab3` an error which is
    measured in another unit than its values, `Mine1999_Tab4` a count which is
    no number and `Mine1999_Tab5` a point without a count. The CV and the
    column without a unit are the two shapes the unit of a column is read in
    (`sbmlsim.serve.data._unit_of`).
    """
    folder = tmp_path / "Mine1999"
    folder.mkdir()
    (folder / ".Mine1999_Tab1.tsv").write_text(
        "label\ttime\ttime_unit\tmean\tmean_cv\tmean_unit\tcount\tratio\n"
        "urine\t12\thr\t3.01\t12.0\tmg\t4\t0.21\n"
        "urine\t96\thr\t4.13\t9.0\tmg\t4\t0.29\n",
        encoding="utf-8",
    )
    (folder / ".Mine1999_Tab3.tsv").write_text(
        "label\ttime\ttime_unit\tmean\tmean_sd\tmean_unit\tmean_sd_unit\n"
        "urine\t12\thr\t3.01\t120.0\tmg\tug\n",
        encoding="utf-8",
    )
    (folder / ".Mine1999_Tab2.tsv").write_text(
        "label\ttime\tmean\tmean_unit\tcount\n"
        "urine\t12\t3.01\tmg\t4\n"
        "urine\t96\t4.13\tmg\t4\n",
        encoding="utf-8",
    )
    (folder / ".Mine1999_Tab4.tsv").write_text(
        "label\ttime\ttime_unit\tmean\tmean_unit\tcount\n"
        "urine\t12\thr\t3.01\tmg\tfour\n",
        encoding="utf-8",
    )
    (folder / ".Mine1999_Tab5.tsv").write_text(
        "label\ttime\ttime_unit\tmean\tmean_unit\tcount\n"
        "urine\t12\thr\t3.01\tmg\t4\n"
        "urine\t96\thr\t4.13\tmg\t\n",
        encoding="utf-8",
    )
    return project.replace(data_path=tmp_path)


def entry_of(project: Project) -> ModelEntry:
    """The model entry of the project, which `panel_data` reads the data of."""
    return ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")


def served(project: Project, panel: Panel) -> TestClient:
    """A client of an application which serves one panel of the project."""
    entry = entry_of(project.replace(panels={panel.id: panel}))
    return TestClient(create_app(ModelRegistry({entry.id: entry})))


def series_of(runner: PanelRunner, project: Project, panel: Panel) -> list[PanelSeries]:
    """Read the study data of a panel in the units the runner resolves."""
    entry = entry_of(project)
    return panel_data(entry, panel, runner.units(entry, panel))


def with_data(panel: Panel, data: PanelData) -> Panel:
    """The panel with one series of study data instead of its own."""
    return panel.model_copy(update={"data": (data,)})


def changed(data: PanelData, **fields: Any) -> PanelData:
    """The series with the fields changed, which the panel of a test declares."""
    return data.model_copy(update=fields)


def in_hours(panel: Panel) -> Panel:
    """The panel with its time axis declared in hours."""
    return panel.model_copy(
        update={"simulation": panel.simulation.model_copy(update={"x_unit": "hr"})}
    )


def test_the_data_of_a_panel_is_served_in_the_unit_of_its_curve(
    client: TestClient,
) -> None:
    body = client.get(DATA).json()
    assert body["contract"] == "paneldata/1"
    assert body["contract"] == DATA_CONTRACT
    series = body["series"][0]
    assert (series["id"], series["label"]) == (URINE.id, URINE.label)
    # the unit of the curve `Aurine_hctz` the series belongs under
    assert series["unit"] == "umol"
    assert series["x_unit"] == X_UNIT
    assert len(series["x"]) == len(series["y"]) == len(MINUTES)
    assert series["x"] == pytest.approx(MINUTES)
    assert series["y"] == pytest.approx(AMOUNTS)
    assert series["count"] == [1, 1, 1, 1]
    assert series["yerr"] is None


def test_a_panel_without_data_answers_an_empty_list_without_its_model(
    project: Project, panel: Panel, never_loads: None
) -> None:
    """There is no unit to resolve, so the answer costs no model."""
    without = panel.model_copy(update={"data": ()})
    assert served(project, without).get(DATA).json() == {
        "contract": DATA_CONTRACT,
        "series": [],
    }


def test_a_dataset_which_does_not_exist_is_reported_as_500_without_a_path(
    project: Project, panel: Panel
) -> None:
    missing = with_data(panel, changed(URINE, dataset="Nowhere_Fig9"))
    response = served(project, missing).get(DATA)
    assert response.status_code == 500
    assert response.json() == {"detail": FAILED}
    for leak in ("Nowhere_Fig9", ".tsv", "examples", "data_path", "Traceback"):
        assert leak not in response.text


def test_a_dataset_name_which_traverses_out_of_the_data_directory_is_refused() -> None:
    """It is refused where the panel is read, so no file is ever resolved."""
    with pytest.raises(ValidationError, match="no dataset"):
        PanelData(
            id="beermann1976",
            label="Beermann 1976",
            dataset="../../etc/passwd",
            x="time",
            y="value",
        )


def test_an_unknown_model_or_panel_is_404(client: TestClient) -> None:
    assert client.get("/api/models/nothing/panels/renal/data").status_code == 404
    assert client.get("/api/models/hctz/panels/nothing/data").status_code == 404


def test_the_x_of_the_data_is_the_x_of_the_simulation_of_the_same_panel(
    client: TestClient,
) -> None:
    """A reader draws the points of a series on the axis of the lines."""
    data = client.get(DATA).json()
    simulated = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    assert data["series"][0]["x_unit"] == simulated["units"]["x"]
    # the dataset is in hours, the x of the panel in the minutes of the model
    assert data["series"][0]["x_unit"] == X_UNIT
    assert data["series"][0]["x"] == pytest.approx(MINUTES)
    # a property of the fixture and of no route: the study reaches 168 hr and
    # the panel simulates a day of it, which a renderer clips
    assert max(data["series"][0]["x"]) > max(simulated["x"])


def test_the_x_of_the_data_is_the_x_unit_the_panel_declares(
    project: Project, panel: Panel
) -> None:
    """The two routes deliver the declared unit, and not one of them alone."""
    client = served(project, in_hours(panel))
    data = client.get(DATA).json()
    simulated = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    assert data["series"][0]["x_unit"] == simulated["units"]["x"] == "hr"
    assert data["series"][0]["x"] == pytest.approx(TIMES)


def test_a_unit_id_of_the_model_is_served_on_both_routes_as_well(
    project: Project, panel: Panel
) -> None:
    """`KI__min` and `substance` are units of the HCTZ model and none of pint."""
    urine = PanelCurve(id="Aurine_hctz", label="urine", unit="substance")
    declared = panel.model_copy(
        update={
            "curves": (panel.curves[0], urine),
            "simulation": panel.simulation.model_copy(update={"x_unit": "KI__min"}),
        }
    )
    client = served(project, declared)
    data = client.get(DATA).json()
    simulated = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    series = data["series"][0]
    assert series["x_unit"] == simulated["units"]["x"] == "KI__min"
    assert series["unit"] == simulated["units"]["Aurine_hctz"] == "substance"
    assert series["x"] == pytest.approx(MINUTES)
    assert series["y"] == pytest.approx([amount * 1e-6 for amount in AMOUNTS])


def test_a_curve_which_declares_no_unit_is_served_in_the_unit_of_the_model(
    project: Project, panel: Panel
) -> None:
    """The curve of a run and its study data are one unit, declared or not."""
    urine = PanelCurve(id="Aurine_hctz", label="urine")
    assert urine.unit is None
    declared = panel.model_copy(update={"curves": (panel.curves[0], urine)})
    client = served(project, declared)
    data = client.get(DATA).json()
    simulated = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    series = data["series"][0]
    # the model carries the amount in the urine in `mmol`, the study in `mg`
    assert series["unit"] == simulated["units"]["Aurine_hctz"] == "mmol"
    assert series["y"] == pytest.approx([amount * 1e-3 for amount in AMOUNTS])


def test_a_curve_the_model_gives_no_unit_is_null_on_both_routes(
    crafted: Project,
) -> None:
    """No unit has one shape on the wire, `null`, and not two texts.

    A renderer compares the unit of a series with the one of the curve it is
    drawn under, so `paneldata/1` and `panel/1` must state it alike. The model
    of this panel declares no units, so its species carry none.
    """
    client = served(crafted.replace(variants={"default": PLAIN_MODEL}), PLAIN_PANEL)
    data = client.get(DATA).json()
    simulated = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    series = data["series"][0]
    assert series["unit"] is None
    assert series["unit"] == simulated["units"]["prey"]
    assert simulated["units"]["prey"] != ""
    # the time of a model without time units is the `second` it falls back to
    assert series["x_unit"] == simulated["units"]["x"] == "second"
    assert series["y"] == pytest.approx(RATIOS)


def test_the_body_of_a_series_is_json_without_a_nan(client: TestClient) -> None:
    """A row of a dataset which carries no point is no point of the series."""
    assert "NaN" not in client.get(DATA).text


def test_the_values_of_a_series_are_converted_before_the_unit_of_its_curve(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    [series] = series_of(runner, project, panel)
    assert (series.unit, series.x_unit) == ("umol", X_UNIT)
    assert series.y == pytest.approx(AMOUNTS)
    assert series.count == [1, 1, 1, 1]


def test_a_series_which_cannot_be_converted_to_the_unit_of_its_curve_is_refused(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    plain = with_data(panel, changed(URINE, conversion=None))
    with pytest.raises(ValueError, match="umol") as error:
        series_of(runner, project, plain)
    message = str(error.value)
    assert "mg" in message
    assert URINE.id in message
    assert ".tsv" not in message


def test_two_series_of_one_curve_are_both_answered_in_its_unit(
    project: Project, panel: Panel
) -> None:
    """A panel shows the subjects of a study, or two studies, under one curve."""
    first = changed(URINE, sel={"label": "amount_urine_hctz5po_1"})
    second = changed(
        URINE,
        id="beermann1976_urine_4",
        label="Beermann 1976 (5 mg, oral, subject 4)",
    )
    both = panel.model_copy(update={"data": (first, second)})
    body = served(project, both).get(DATA).json()
    assert [series["id"] for series in body["series"]] == [first.id, second.id]
    assert [series["unit"] for series in body["series"]] == ["umol", "umol"]
    assert [series["x_unit"] for series in body["series"]] == [X_UNIT, X_UNIT]
    assert body["series"][1]["y"] == pytest.approx(AMOUNTS)
    assert body["series"][0]["y"] != body["series"][1]["y"]


def test_the_unit_a_series_declares_wins_over_the_one_of_its_curve(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    declared = with_data(panel, changed(URINE, unit="nmol"))
    [series] = series_of(runner, project, declared)
    assert series.unit == "nmol"
    assert series.y == pytest.approx([amount * 1e3 for amount in AMOUNTS])


def test_a_series_which_names_no_curve_keeps_the_unit_of_its_dataset(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    own = with_data(panel, changed(URINE, curve=None, conversion=None))
    [series] = series_of(runner, project, own)
    assert series.unit == "mg"
    assert series.y == pytest.approx(MASSES)


def test_the_times_of_a_series_are_the_x_unit_a_panel_declares(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    """The `x_unit` of a panel is the one its curves are delivered in as well."""
    [series] = series_of(runner, project, in_hours(panel))
    assert series.x_unit == "hr"
    assert series.x == pytest.approx(TIMES)


def test_the_error_and_the_count_of_a_series_are_the_ones_of_its_study(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    [series] = series_of(runner, project, with_data(panel, PATEL))
    assert series.unit == "uM"
    assert len(series.x) == len(series.y) == 11
    assert series.y[0] == pytest.approx(22.8 / 297.7)
    assert series.yerr is not None
    assert len(series.yerr) == 11
    assert series.yerr[0] == pytest.approx(21.9 / 297.7)
    assert series.count == [12] * 11


def test_a_row_without_a_value_is_no_point_of_the_series(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    """The group data of Patel 1984 has its mean and no value of a subject."""
    values = with_data(panel, changed(PATEL, y="value", yerr=None))
    [series] = series_of(runner, project, values)
    assert (series.x, series.y, series.count) == ([], [], None)


def test_a_count_which_is_no_number_is_refused_naming_the_series(
    runner: PanelRunner, crafted: Project, panel: Panel
) -> None:
    """The counts are numbers of subjects, a text of the column is a problem."""
    declared = with_data(panel, changed(MINE, dataset="Mine1999_Tab4"))
    with pytest.raises(ValueError, match="count") as error:
        series_of(runner, crafted, declared)
    message = str(error.value)
    assert MINE.id in message
    assert "Mine1999_Tab4" in message
    assert ".tsv" not in message


def test_a_series_whose_points_do_not_all_carry_a_count_carries_none(
    runner: PanelRunner,
    crafted: Project,
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A point without a count is no count of one, which is reported."""
    declared = with_data(panel, changed(MINE, dataset="Mine1999_Tab5"))
    with caplog.at_level(logging.WARNING, logger="sbmlsim.serve.data"):
        [series] = series_of(runner, crafted, declared)
    assert series.count is None
    assert len(series.y) == 2
    assert caplog.records
    for named in (MINE.id, "Mine1999_Tab5", "count"):
        assert named in caplog.text


def test_a_relative_cv_is_no_error_bar_of_a_series(
    runner: PanelRunner, crafted: Project, panel: Panel
) -> None:
    """A `_cv` is relative, so it is no quantity in the unit of its column."""
    relative = with_data(panel, changed(MINE, yerr="mean_cv"))
    with pytest.raises(ValueError, match="mean_cv") as error:
        series_of(runner, crafted, relative)
    message = str(error.value)
    assert MINE.id in message
    assert "no unit" in message


def test_an_error_measured_in_another_unit_is_delivered_in_the_one_of_its_values(
    runner: PanelRunner, crafted: Project, panel: Panel
) -> None:
    """The dataset measures the mean in `mg` and its deviation in `ug`."""
    own = with_data(
        panel,
        PanelData(
            id="mine1999",
            label="Mine 1999",
            dataset="Mine1999_Tab3",
            x="time",
            y="mean",
            yerr="mean_sd",
        ),
    )
    [series] = series_of(runner, crafted, own)
    assert series.unit == "mg"
    assert series.y == pytest.approx([3.01])
    assert series.yerr == pytest.approx([0.12])


def test_the_times_of_a_dataset_without_a_unit_are_delivered_as_they_are(
    runner: PanelRunner, crafted: Project, panel: Panel
) -> None:
    """A model whose time has no unit and a dataset which has none agree."""
    declared = with_data(panel, PLAIN)
    entry = entry_of(crafted)
    units = runner.units(entry, declared)
    [series] = panel_data(entry, declared, PanelUnits(x=UNITLESS, series=units.series))
    assert series.x_unit is None
    assert series.x == pytest.approx(MINE_TIMES)
    assert series.unit == "umol"


def test_a_curve_the_model_gives_no_unit_delivers_the_values_as_they_are(
    runner: PanelRunner, crafted: Project, panel: Panel
) -> None:
    """A curve without a unit and a column without one are one scale."""
    declared = with_data(panel, changed(MINE, y="ratio", conversion=None))
    entry = entry_of(crafted)
    units = runner.units(entry, declared)
    [series] = panel_data(
        entry, declared, PanelUnits(x=units.x, series={MINE.id: UNITLESS})
    )
    assert series.unit is None
    assert series.y == pytest.approx(RATIOS)
    assert (series.x_unit, series.x) == (X_UNIT, pytest.approx(MINE_MINUTES))


def test_values_which_carry_a_unit_under_a_curve_without_one_are_refused(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    """The study is a mass and the curve of the model carries no unit."""
    entry = entry_of(project)
    units = runner.units(entry, panel)
    with pytest.raises(ValueError, match="mg") as error:
        panel_data(entry, panel, PanelUnits(x=units.x, series={URINE.id: UNITLESS}))
    message = str(error.value)
    assert URINE.id in message
    assert "no one scale" in message


def test_the_times_of_a_dataset_a_unitless_model_cannot_draw_are_refused(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    """The times of the study are hours and the time of the model has no unit."""
    entry = entry_of(project)
    units = runner.units(entry, panel)
    with pytest.raises(ValueError, match="hr") as error:
        panel_data(entry, panel, PanelUnits(x=UNITLESS, series=units.series))
    message = str(error.value)
    assert URINE.id in message
    assert ".tsv" not in message


def test_a_selection_which_no_row_has_is_refused_without_a_path(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, sel={"label": "nothing"}))
    with pytest.raises(ValueError, match="nothing") as error:
        series_of(runner, project, other)
    message = str(error.value)
    assert URINE.dataset in message
    assert ".tsv" not in message
    response = served(project, other).get(DATA)
    assert response.status_code == 500
    assert response.json() == {"detail": FAILED}


def test_a_selection_of_a_column_the_dataset_has_not_is_refused(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, sel={"nothing": "x"}))
    with pytest.raises(ValueError, match="nothing") as error:
        series_of(runner, project, other)
    assert URINE.dataset in str(error.value)


def test_a_column_the_dataset_has_not_is_refused(
    runner: PanelRunner, project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, y="nothing"))
    with pytest.raises(ValueError, match="nothing") as error:
        series_of(runner, project, other)
    message = str(error.value)
    assert URINE.id in message
    assert ".tsv" not in message
