"""The study data behind the curves of a panel.

The model, the panel and the application of the tests are the fixtures of
`tests/serve/conftest.py`: the panel declares the cumulative amount of
hydrochlorothiazide in the urine of Beermann 1976 under its curve of the
urine, measured in mg and delivered in the `umol` of that curve.

A panel of another declaration is served by an application of its own
(`served`), and the series themselves are read with `panel_data`, which needs
no application at all.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sbmlsim.panel import Panel, PanelData
from sbmlsim.project import Project
from sbmlsim.serve.app import FAILED, create_app
from sbmlsim.serve.data import panel_data
from sbmlsim.serve.payload import DATA_CONTRACT
from sbmlsim.serve.runner import ModelEntry, ModelRegistry
from tests.serve.conftest import DATA, MR_HCTZ, URINE

#: the times of the series of the panel, in the `hr` of its dataset
TIMES = [0.0, 12.0, 96.0, 168.0]
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


def entry_of(project: Project) -> ModelEntry:
    """The model entry of the project, which `panel_data` reads the data of."""
    return ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")


def served(project: Project, panel: Panel) -> TestClient:
    """A client of an application which serves one panel of the project."""
    entry = entry_of(project.replace(panels={panel.id: panel}))
    return TestClient(create_app(ModelRegistry({entry.id: entry})))


def with_data(panel: Panel, data: PanelData) -> Panel:
    """The panel with one series of study data instead of its own."""
    return panel.model_copy(update={"data": (data,)})


def changed(data: PanelData, **fields: Any) -> PanelData:
    """The series with the fields changed, which the panel of a test declares."""
    return data.model_copy(update=fields)


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
    assert series["x_unit"] == "hr"
    assert len(series["x"]) == len(series["y"]) == len(TIMES)
    assert series["x"] == pytest.approx(TIMES)
    assert series["y"] == pytest.approx(AMOUNTS)
    assert series["count"] == [1, 1, 1, 1]
    assert series["yerr"] is None


def test_a_panel_without_data_answers_an_empty_series_list(
    project: Project, panel: Panel
) -> None:
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


def test_the_data_of_a_panel_is_read_without_loading_a_model(
    client: TestClient, never_loads: None
) -> None:
    """The data of a panel is read from the files of its project alone."""
    assert client.get(DATA).status_code == 200


def test_the_body_of_a_series_is_json_without_a_nan(client: TestClient) -> None:
    """A row of a dataset which carries no point is no point of the series."""
    assert "NaN" not in client.get(DATA).text


def test_the_values_of_a_series_are_converted_before_the_unit_of_its_curve(
    project: Project, panel: Panel
) -> None:
    [series] = panel_data(entry_of(project), panel)
    assert (series.unit, series.x_unit) == ("umol", "hr")
    assert series.y == pytest.approx(AMOUNTS)
    assert series.count == [1, 1, 1, 1]


def test_a_series_which_cannot_be_converted_to_the_unit_of_its_curve_is_refused(
    project: Project, panel: Panel
) -> None:
    plain = with_data(panel, changed(URINE, conversion=None))
    with pytest.raises(ValueError, match="umol") as error:
        panel_data(entry_of(project), plain)
    message = str(error.value)
    assert "mg" in message
    assert URINE.id in message
    assert ".tsv" not in message


def test_the_unit_a_series_declares_wins_over_the_one_of_its_curve(
    project: Project, panel: Panel
) -> None:
    declared = with_data(panel, changed(URINE, unit="nmol"))
    [series] = panel_data(entry_of(project), declared)
    assert series.unit == "nmol"
    assert series.y == pytest.approx([amount * 1e3 for amount in AMOUNTS])


def test_a_series_which_names_no_curve_keeps_the_unit_of_its_dataset(
    project: Project, panel: Panel
) -> None:
    own = with_data(panel, changed(URINE, id="beermann1976", conversion=None))
    [series] = panel_data(entry_of(project), own)
    assert series.unit == "mg"
    assert series.y == pytest.approx(MASSES)


def test_the_times_of_a_series_are_the_x_unit_of_the_panel(
    project: Project, panel: Panel
) -> None:
    """The x unit of a panel is the one its curves are delivered in as well."""
    minutes = panel.simulation.model_copy(update={"x_unit": "min"})
    [series] = panel_data(
        entry_of(project), panel.model_copy(update={"simulation": minutes})
    )
    assert series.x_unit == "min"
    assert series.x == pytest.approx([time * 60.0 for time in TIMES])


def test_the_error_and_the_count_of_a_series_are_the_ones_of_its_study(
    project: Project, panel: Panel
) -> None:
    [series] = panel_data(entry_of(project), with_data(panel, PATEL))
    assert series.unit == "uM"
    assert len(series.x) == len(series.y) == 11
    assert series.y[0] == pytest.approx(22.8 / 297.7)
    assert series.yerr is not None
    assert len(series.yerr) == 11
    assert series.yerr[0] == pytest.approx(21.9 / 297.7)
    assert series.count == [12] * 11


def test_a_row_without_a_value_is_no_point_of_the_series(
    project: Project, panel: Panel
) -> None:
    """The group data of Patel 1984 has its mean and no value of a subject."""
    values = with_data(panel, changed(PATEL, y="value", yerr=None))
    [series] = panel_data(entry_of(project), values)
    assert (series.x, series.y, series.count) == ([], [], None)


def test_a_selection_which_no_row_has_is_refused_without_a_path(
    project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, sel={"label": "nothing"}))
    with pytest.raises(ValueError, match="nothing") as error:
        panel_data(entry_of(project), other)
    message = str(error.value)
    assert URINE.dataset in message
    assert ".tsv" not in message
    response = served(project, other).get(DATA)
    assert response.status_code == 500
    assert response.json() == {"detail": FAILED}


def test_a_selection_of_a_column_the_dataset_has_not_is_refused(
    project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, sel={"nothing": "x"}))
    with pytest.raises(ValueError, match="nothing") as error:
        panel_data(entry_of(project), other)
    assert URINE.dataset in str(error.value)


def test_a_column_the_dataset_has_not_is_refused(
    project: Project, panel: Panel
) -> None:
    other = with_data(panel, changed(URINE, y="nothing"))
    with pytest.raises(ValueError, match="nothing") as error:
        panel_data(entry_of(project), other)
    message = str(error.value)
    assert URINE.id in message
    assert ".tsv" not in message
