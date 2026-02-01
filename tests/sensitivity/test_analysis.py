from dataclasses import asdict

from sbmlsim.sensitivity import SensitivityOutput, AnalysisGroup


# -----------------------------------------------------------------------------
# SensitivityOutput
# -----------------------------------------------------------------------------


def test_sensitivity_output_creation() -> None:
    out = SensitivityOutput(
        uid="auc_plasma",
        name="AUC plasma",
        unit="mg*h/L",
    )

    assert out.uid == "auc_plasma"
    assert out.name == "AUC plasma"
    assert out.unit == "mg*h/L"


def test_sensitivity_output_unit_optional() -> None:
    out = SensitivityOutput(
        uid="cmax",
        name="Cmax",
        unit=None,
    )

    assert out.unit is None


def test_sensitivity_output_equality() -> None:
    o1 = SensitivityOutput("auc", "AUC", "mg*h/L")
    o2 = SensitivityOutput("auc", "AUC", "mg*h/L")

    assert o1 == o2


def test_sensitivity_output_asdict() -> None:
    out = SensitivityOutput("auc", "AUC", "mg*h/L")

    d = asdict(out)

    assert d == {
        "uid": "auc",
        "name": "AUC",
        "unit": "mg*h/L",
    }


# -----------------------------------------------------------------------------
# AnalysisGroup
# -----------------------------------------------------------------------------


def test_analysis_group_creation() -> None:
    group = AnalysisGroup(
        uid="renal_impairment",
        name="Renal impairment",
        changes={"GFR": 0.5, "CLr": 0.6},
        color="blue",
    )

    assert group.uid == "renal_impairment"
    assert group.name == "Renal impairment"
    assert group.changes == {"GFR": 0.5, "CLr": 0.6}
    assert group.color == "blue"


def test_analysis_group_color_optional() -> None:
    group = AnalysisGroup(
        uid="baseline",
        name="Baseline",
        changes={},
        color=None,
    )

    assert group.color is None


def test_analysis_group_changes_mutable() -> None:
    group = AnalysisGroup(
        uid="test",
        name="Test",
        changes={"k1": 1.0},
        color=None,
    )

    group.changes["k2"] = 2.0

    assert group.changes == {"k1": 1.0, "k2": 2.0}


def test_analysis_group_equality() -> None:
    g1 = AnalysisGroup(
        uid="hepatic",
        name="Hepatic impairment",
        changes={"CL": 0.7},
        color="red",
    )
    g2 = AnalysisGroup(
        uid="hepatic",
        name="Hepatic impairment",
        changes={"CL": 0.7},
        color="red",
    )

    assert g1 == g2


def test_analysis_group_asdict() -> None:
    group = AnalysisGroup(
        uid="dose_up",
        name="Dose increase",
        changes={"Dose": 2.0},
        color="green",
    )

    d = asdict(group)

    assert d == {
        "uid": "dose_up",
        "name": "Dose increase",
        "changes": {"Dose": 2.0},
        "color": "green",
    }
