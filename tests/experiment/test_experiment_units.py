"""The quantities of an experiment resolve the unit ids of its models (#268)."""

from pathlib import Path
from typing import override

import libsbml
import pytest
from pint.errors import UndefinedUnitError

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml

#: the units of the unit definitions, `(kind, multiplier, scale, exponent)`
MILLIGRAM = (libsbml.UNIT_KIND_GRAM, 1.0, -3, 1)
GRAM = (libsbml.UNIT_KIND_GRAM, 1.0, 0, 1)
MILLIMOLE = (libsbml.UNIT_KIND_MOLE, 1.0, -3, 1)
PER_MINUTE = (libsbml.UNIT_KIND_SECOND, 60.0, 0, -1)
PER_HOUR = (libsbml.UNIT_KIND_SECOND, 3600.0, 0, -1)


def _probe(
    path: Path, f_unit: str, **definitions: tuple[tuple[int, float, int, int], ...]
) -> Path:
    """Write the probe model with the unit definitions and the unit of `f`."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml())
    model: libsbml.Model = doc.getModel()
    for uid, units in definitions.items():
        udef: libsbml.UnitDefinition = model.createUnitDefinition()
        udef.setId(uid)
        for kind, multiplier, scale, exponent in units:
            unit: libsbml.Unit = udef.createUnit()
            unit.setKind(kind)
            unit.setMultiplier(multiplier)
            unit.setScale(scale)
            unit.setExponent(exponent)
    model.getParameter("f").setUnits(f_unit)
    path.write_text(libsbml.writeSBMLToString(doc))
    return path


def _experiment(
    models: dict[str, Path], simulate: bool = False
) -> type[SimulationExperiment]:
    """Get an experiment of the models, which sets `f` with a unit id if it simulates."""

    class UnitIds(SimulationExperiment):
        @override
        def models(self) -> dict[str, AbstractModel | Path]:
            return {mid: AbstractModel(source=path) for mid, path in models.items()}

        @override
        def simulations(self) -> dict[str, Simulation | Scan]:
            if not simulate:
                return {}
            # 0.18 g/hr are 3 mg/min, the unit of `f`
            dose = self.Q(0.18, "g_per_hr")
            return {"s": Simulation(end=1, steps=2, preinit_changes={"f": dose})}

        @override
        def tasks(self) -> dict[str, Task]:
            if "s" not in self._simulations:
                return {}
            return {f"t_{mid}": Task(model=mid, simulation="s") for mid in self._models}

    return UnitIds


def _runner(experiment: type[SimulationExperiment], path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        [experiment], base_path=path, data_path=path, on_error="raise"
    )


def test_an_experiment_simulates_with_a_unit_id_of_its_model(tmp_path: Path) -> None:
    """`self.Q` in `simulations()` takes a unit id of the model, which the runner loaded."""
    path = _probe(
        tmp_path / "m.xml",
        "mg_per_min",
        mg_per_min=(MILLIGRAM, PER_MINUTE),
        g_per_hr=(GRAM, PER_HOUR),
    )
    runner = _runner(_experiment({"m": path}, simulate=True), tmp_path)
    exp = runner.experiments["UnitIds"]
    exp.run(Simulator(), reduced_selections=False)

    result = exp.results["t_m"]
    # kk := 3*f with f = 3 mg/min
    assert result["f"].values[0] == pytest.approx(3.0)
    assert result["kk"].values[0] == pytest.approx(9.0)
    # a unit id with units of pint, and units of pint only
    assert exp.Q(3.0, "mg_per_min/kg").to("mg/min/kg").magnitude == pytest.approx(3.0)
    assert exp.Q(3.0, "mg/min").to("g/hr").magnitude == pytest.approx(0.18)


def test_a_unit_id_of_several_models_must_agree(tmp_path: Path) -> None:
    """Without `model=` a unit id must mean the same in every model which has it."""
    a = _probe(
        tmp_path / "a.xml",
        "u1",
        u1=(MILLIGRAM,),
        g_per_hr=(GRAM, PER_HOUR),
        only_a=(PER_MINUTE,),
    )
    b = _probe(tmp_path / "b.xml", "u1", u1=(MILLIMOLE,), g_per_hr=(GRAM, PER_HOUR))
    runner = _runner(_experiment({"a": a, "b": b}), tmp_path)
    exp = runner.experiments["UnitIds"]

    with pytest.raises(ValueError, match=r"'u1'.*'a'.*'mg'.*'b'.*'mmol'.*model="):
        exp.Q(1.0, "u1/ml")
    assert exp.Q(1.0, "u1", model="a").to("mg").magnitude == pytest.approx(1.0)
    assert exp.Q(1.0, "u1", model="b").to("mmol").magnitude == pytest.approx(1.0)
    # an id of one model, an id with one expression in both, and units of pint
    assert exp.Q(1.0, "only_a").to("1/min").magnitude == pytest.approx(1.0)
    assert exp.Q(1.0, "g_per_hr").to("g/hr").magnitude == pytest.approx(1.0)
    assert exp.Q(1.0, "mmol").to("mol").magnitude == pytest.approx(1e-3)
    # a unit of pint is taken as it is, also with `model=`
    unit = exp.Q(1.0, "only_a").units
    assert exp.Q(2.0, unit) == exp.Q(2.0, unit, model="b") == exp.Q(2.0, "1/min")
    with pytest.raises(ValueError, match=r"'c'.*\['a', 'b'\]"):
        exp.Q(1.0, "mg", model="c")
    with pytest.raises(UndefinedUnitError) as info:
        exp.Q(1.0, "u2")
    assert "'a', 'b'" in "\n".join(info.value.__notes__)


def test_an_experiment_without_loaded_models_takes_pint_units(tmp_path: Path) -> None:
    """Before the runner loaded the models a unit id is unknown, which the error says."""
    path = _probe(tmp_path / "m.xml", "mg_per_min", mg_per_min=(MILLIGRAM, PER_MINUTE))
    exp = _experiment({"m": path})(base_path=tmp_path, data_path=tmp_path)

    assert exp.Q(3.0, "mg/min").to("g/hr").magnitude == pytest.approx(0.18)
    with pytest.raises(UndefinedUnitError) as info:
        exp.Q(3.0, "mg_per_min")
    assert "not loaded" in "\n".join(info.value.__notes__)
    with pytest.raises(ValueError, match=r"'m'.*not loaded"):
        exp.Q(3.0, "mg_per_min", model="m")
