"""The runner of the sensitivity analyses on the HCTZ model: the pharmacokinetics of an oral dose.

The project of the module has the HCTZ model as its one variant and the runner
`pk`, which varies the dissolution and the renal excretion of HCTZ and reads
the non-compartmental analysis of the plasma concentration (`PK`), whose
observables the runner builds when it runs.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from examples.hctz_fitting import HCTZ_PATH, MODEL_PATH
from examples.hctz_fitting.project import observables, oral_dose
from sbmlsim import run
from sbmlsim.project import Project, SimulatorSettings
from sbmlsim.sensitivity.parameters import (
    ParameterType,
    SensitivityParameter,
    parameters_from_sbml,
    set_bounds,
    set_relative_bounds,
)
from sbmlsim.sensitivity.runner import (
    LocalAnalysis,
    SamplingAnalysis,
    SensitivityGroup,
    SensitivityOutput,
    SensitivityRunner,
)

#: the parameters of the analysis
PARAMETER_IDS = ("Ka_dis_hctz", "KI__HCTZEX_k")


def hctz_parameters() -> list[SensitivityParameter]:
    """Get the dissolution and the renal excretion, +-50 %, the excretion fitted."""
    parameters = [
        p
        for p in parameters_from_sbml(MODEL_PATH, exclude_zero=False)
        if p.uid in PARAMETER_IDS
    ]
    set_bounds(parameters, [("KI__HCTZEX_k", 0.0005, 0.0015, ParameterType.FIT)])
    set_relative_bounds(parameters, 0.5)
    return parameters


ORAL = SensitivityGroup("oral", "Oral dose", oral_dose())

PROJECT = Project(
    name="hctz",
    root=HCTZ_PATH,
    variants={"default": MODEL_PATH},
    simulator=SimulatorSettings(jit_optimization="O0"),
    sensitivities={
        "pk": SensitivityRunner(
            parameters=hctz_parameters,
            outputs=[
                SensitivityOutput("cmax", "HCTZ Cmax", observable="pk.cmax"),
                SensitivityOutput("auc", "HCTZ AUC", observable="pk.auc_inf_obs"),
            ],
            observables=observables,
            analyses={
                "local": LocalAnalysis(groups=[ORAL], n_var=1, delta=0.1),
                "sampling": SamplingAnalysis(groups=[ORAL], n=4),
            },
            default=("local", "sampling"),
        )
    },
)
SPEC: dict[str, Any] = {"project": f"{__name__}:PROJECT", "analysis": "pk"}


def test_the_pharmacokinetics_of_an_oral_dose(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute("sensitivity", SPEC, out, workers=1)
    assert summary == {"analyses": ["local", "sampling"], "parameters": 2}
    table = pd.read_csv(out / "parameters.tsv", sep="\t").set_index("uid")
    assert list(table.index) == ["KI__HCTZEX_k", "Ka_dis_hctz"]
    assert table.loc["KI__HCTZEX_k", "type"] == "fitted"
    assert table.loc["KI__HCTZEX_k", "lower_bound"] == 0.0005
    local = pd.read_csv(out / "local" / "local_d0.1_nvar1_00_oral.tsv", sep="\t")
    assert set(local["output"]) == {"cmax", "auc"}
    # the renal excretion lowers the exposure
    auc = local[(local["output"] == "auc") & (local["parameter"] == "KI__HCTZEX_k")]
    assert auc["Sij"].iloc[0] < 0
    sampling = pd.read_csv(out / "sampling" / "sampling_N4.tsv", sep="\t")
    assert sampling["uid"].tolist() == ["cmax", "auc"]
    assert (sampling["mean"] > 0).all()
    samples = pd.read_csv(out / "sampling" / "oral" / "samples.tsv", sep="\t")
    assert samples["KI__HCTZEX_k"].between(0.0005, 0.0015).all()


def test_the_runner_lists_its_analyses() -> None:
    (entry,) = [a for a in PROJECT.analyses() if a["kind"] == "sensitivity"]
    assert entry == {
        "kind": "sensitivity",
        "name": "pk",
        "description": "the sensitivity analyses 'pk': local, sampling",
    }


@pytest.mark.parametrize("value", ["2"])
def test_a_smoke_run_of_the_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SBMLSIM_SENSITIVITY_N", value)
    out = tmp_path / "out"
    run.execute("sensitivity", {**SPEC, "analyses": ["sampling"]}, out, workers=1)
    assert (out / "sampling" / "sampling_N2.tsv").is_file()
