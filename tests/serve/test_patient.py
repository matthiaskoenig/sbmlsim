"""The page `patient` of the HCTZ demo and its project of two variants.

`examples/panels/hctz.py` and `examples/panels/hctz_variants.py` are what the
end to end tests of sbmlsim-web serve, so
these tests hold it to every feature it is there to exercise: a knob without a
target read by the formula of a derived target, the named stops of a range
knob and a series of the study data measured at one, the fixed limits of the
axes, the scenarios of a page and, with `PROJECT_VARIANTS`, a declaration
which differs by the variant.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from examples.panels import hctz, hctz_variants
from examples.panels.hctz import (
    HCTZ_PATIENT_PLASMA,
    HCTZ_PATIENT_URINE,
    PATIENT_PLASMA_MAX,
    PATIENT_URINE_MAX,
    PATIENT_X_MAX,
    PROJECT,
)
from examples.panels.hctz_variants import LOW_REFERENCE
from sbmlsim.panel import Panel
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner

#: the root of the repository, which the package `examples` is imported from
ROOT = Path(__file__).parents[2]
#: the routes of the page `patient`
PLASMA = "/api/models/hctz/panels/patient_plasma"
URINE = "/api/models/hctz/panels/patient_urine"


@pytest.fixture(scope="module")
def variants() -> Iterator[TestClient]:
    """A client of `PROJECT_VARIANTS`, registered as a service registers it."""
    registry = ModelRegistry.of(
        {"hctz": "examples.panels.hctz_variants:PROJECT_VARIANTS"}
    )
    with TestClient(create_app(registry)) as client:
        yield client


def test_the_demo_passes_the_checks_of_the_startup_of_a_service() -> None:
    registry = ModelRegistry.of({"hctz": "examples.panels.hctz:PROJECT"})
    assert registry.panel("hctz", "patient_plasma") == HCTZ_PATIENT_PLASMA


def test_the_page_patient_declares_every_feature_of_the_extension(
    client: TestClient,
) -> None:
    panel = client.get(PLASMA).json()
    crcl = panel["axes"][0]
    assert (crcl["id"], crcl["target"], crcl["unit"]) == ("crcl", None, "ml/min")
    assert crcl["kind"] == "range"
    assert [stop["value"] for stop in crcl["choices"]] == [
        110.0,
        75.0,
        45.0,
        20.0,
        10.0,
    ]
    assert panel["simulation"]["derived"] == {"f_renal_function": "min(crcl / 110, 1)"}
    assert panel["simulation"]["x_unit"] == "hr"
    assert {key: panel["panel"][key] for key in ("y_min", "y_max", "x_max")} == {
        "y_min": 0.0,
        "y_max": PATIENT_PLASMA_MAX,
        "x_max": PATIENT_X_MAX,
    }
    app = client.get("/api/models/hctz/app").json()
    # the knobs of both panels are shared, and the scenarios set both
    assert [one["knob"]["id"] for one in app["shared"]["patient"]] == ["crcl", "dose"]
    assert app["pages"][2]["scenarios"][2] == {
        "id": "severe",
        "label": "severe impairment, 12.5 mg",
        "values": {"crcl": 20.0, "dose": 12.5},
    }


def test_the_study_data_of_the_page_is_measured_at_a_stop(client: TestClient) -> None:
    """Beermann 1976 was a healthy volunteer, the stop `normal`, after 5 mg."""
    panel = client.get(URINE).json()
    assert panel["data"][0]["at"] == {"crcl": 110.0, "dose": 5.0}
    data = client.get(f"{URINE}/data").json()["series"]
    assert len(data) == 1
    # the times of the study are delivered in the hours of the axis
    assert data[0]["x_unit"] == "hr"
    assert data[0]["x"][:2] == [0.0, 12.0]


@pytest.mark.parametrize(
    ("panel", "limit"),
    [
        (HCTZ_PATIENT_PLASMA, PATIENT_PLASMA_MAX),
        (HCTZ_PATIENT_URINE, PATIENT_URINE_MAX),
    ],
)
def test_the_fixed_value_axis_holds_the_highest_curve_the_knobs_reach(
    runner: PanelRunner, panel: Panel, limit: float
) -> None:
    """The measurement of the comment of the limits, so it stays true."""
    result = runner.run("hctz", panel, {}, {"crcl": 8, "dose": 4})
    assert result.failed == ()
    highest = max(np.max(rows) for rows in result.curves.values())
    assert 0.75 * limit < highest <= limit
    assert result.x[-1] == pytest.approx(PATIENT_X_MAX)


def test_a_scenario_runs_the_renal_function_of_its_clearance(
    client: TestClient,
) -> None:
    """A severe impairment, 20 ml/min, is a renal function of 20 / 110."""
    severe = client.post(
        f"{PLASMA}/simulate", json={"values": {"crcl": 20.0, "dose": 12.5}, "grid": {}}
    ).json()
    renal = client.post(
        "/api/models/hctz/panels/renal_plasma/simulate",
        json={"values": {"egfr": 0.4, "dose": 12.5}, "grid": {}},
    ).json()
    healthy = client.post(
        f"{PLASMA}/simulate", json={"values": {"crcl": 110.0, "dose": 12.5}, "grid": {}}
    ).json()
    peak = {
        name: max(body["curves"]["[Cve_hctz]"][0])
        for name, body in (("severe", severe), ("renal", renal), ("healthy", healthy))
    }
    # 20 / 110 is a worse kidney than 0.4, which is worse than a healthy one
    assert peak["severe"] > peak["renal"] > peak["healthy"]


def test_the_variants_of_the_demo_declare_the_page_otherwise(
    variants: TestClient,
) -> None:
    default = variants.get(PLASMA).json()
    low = variants.get(PLASMA, params={"variant": LOW_REFERENCE}).json()
    assert default["simulation"]["derived"] == {
        "f_renal_function": "min(crcl / 110, 1)"
    }
    assert low["simulation"]["derived"] == {"f_renal_function": "min(crcl / 90, 1)"}
    assert (
        low["panel"]["title"]
        == "Plasma concentration of a patient (reference 90 ml/min)"
    )
    app = variants.get("/api/models/hctz/app", params={"variant": LOW_REFERENCE}).json()
    assert [one["title"] for one in app["heads"]["patient"]] == [
        "Plasma concentration of a patient (reference 90 ml/min)",
        "Urinary excretion of a patient (reference 90 ml/min)",
    ]
    assert [variant["id"] for variant in app["variants"]] == ["default", LOW_REFERENCE]


def test_a_simulation_of_a_variant_of_the_demo_runs_its_renal_function(
    variants: TestClient,
) -> None:
    """45 ml/min over 90 is the renal function of 55 ml/min over 110."""

    def peak(values: dict[str, float], variant: str | None = None) -> Any:
        body: dict[str, Any] = {"values": values, "grid": {}}
        if variant is not None:
            body["variant"] = variant
        response = variants.post(f"{PLASMA}/simulate", json=body).json()
        return np.asarray(response["curves"]["[Cve_hctz]"][0])

    low = peak({"crcl": 45.0}, LOW_REFERENCE)
    default = peak({"crcl": 45.0})
    assert not np.allclose(low, default)
    # the parser of PEtab reads 90 as a float of 15 digits, so the renal
    # function is 0.5 to a few units in the last place and the curves agree
    # within the tolerances of the integrator
    assert np.allclose(low, peak({"crcl": 55.0}), rtol=1e-6, atol=1e-12)


#: the script of `test_the_project_of_the_variants_is_made_on_its_first_use`:
#: the device of the copy before and after `PROJECT_VARIANTS` is asked for
FIRST_USE = """
import json, pathlib, tempfile
import examples.panels.hctz_variants as variants

def directories():
    root = pathlib.Path(tempfile.gettempdir())
    return sorted(str(path) for path in root.glob(variants.DIRECTORY_PREFIX + "*"))

before = [variants._variants_project.cache_info().currsize, directories()]
project = variants.PROJECT_VARIANTS
after = [variants._variants_project.cache_info().currsize, directories()]
copy = str(project.variants[variants.LOW_REFERENCE])
print(json.dumps({"before": before, "after": after, "copy": copy}))
"""


def test_the_project_of_the_variants_is_made_on_its_first_use(tmp_path: Path) -> None:
    """Importing the module writes no copy, asking for the project writes one.

    The process is a fresh interpreter whose temporary directory is `tmp_path`,
    so nothing of this one or of another test is seen, and the copy is removed
    when the process ends.
    """
    environment = {**os.environ, "TMPDIR": str(tmp_path)}
    completed = subprocess.run(
        [sys.executable, "-c", FIRST_USE],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    seen = json.loads(completed.stdout.strip().splitlines()[-1])
    assert seen["before"] == [0, []]
    count, directories = seen["after"]
    assert count == 1
    assert len(directories) == 1
    assert Path(seen["copy"]).parent == Path(directories[0])
    # the process removed its copy when it ended
    assert not Path(directories[0]).exists()
    assert list(tmp_path.iterdir()) == []


def test_the_documented_example_declares_and_copies_no_model() -> None:
    """The copy is a device of `hctz_variants` and no part of the example to copy."""
    assert not hasattr(hctz, "PROJECT_VARIANTS")
    assert not hasattr(hctz, "__getattr__")
    assert "tempfile" not in Path(hctz.__file__).read_text(encoding="utf-8")


def test_the_copy_of_the_variants_is_the_model_and_no_other_name_is_made() -> None:
    copy = hctz_variants.PROJECT_VARIANTS.variants[LOW_REFERENCE]
    assert hctz_variants.PROJECT_VARIANTS is hctz_variants.PROJECT_VARIANTS
    assert copy.read_bytes() == PROJECT.variants["default"].read_bytes()
    with pytest.raises(AttributeError, match="PROJECT_VARIANTZ"):
        _ = hctz_variants.PROJECT_VARIANTZ


@pytest.fixture(scope="module")
def runner(registry: ModelRegistry) -> PanelRunner:
    return PanelRunner(registry)


def test_the_entry_of_the_demo_serves_the_patient_page(registry: ModelRegistry) -> None:
    entry: ModelEntry = registry.entry("hctz")
    assert [panel.id for panel in entry.pages()["patient"]] == [
        "patient_plasma",
        "patient_urine",
    ]
