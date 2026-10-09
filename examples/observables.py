"""Observables of a scan: the plasma concentration of midazolam for three doses.

A Formula gives the mass concentration from the molar one, a PK observable
its non-compartmental analysis with pkpdutils, a Custom the time above a
threshold; the result keeps the values per dose and the timecourses.
"""

from typing import Any

import numpy as np

from sbmlsim import Q
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Custom, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator

#: the threshold of `time_above`, 25 ng/ml in mg/l, the natural unit of the formula
THRESHOLD = 0.025


def time_above(time: np.ndarray, values: dict[str, Any]) -> float:
    """Get the time the plasma concentration is above the threshold.

    A function sees the natural unit of the formula `mid`, which is the unit
    of `[Cve_mid] * Mr_mid`, i.e. mg/l (the declared unit ng/ml only converts
    the result), and the time in the time unit of the model, minutes for
    midazolam.
    """
    above = values["mid"] > THRESHOLD
    return float(np.sum(np.diff(time)[above[:-1]]))


def run() -> Any:
    """Run the scan over the dose and print the PK parameters per dose."""
    simulation = Simulation(
        time_unit="hr",
        end=24,
        steps=480,
        changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
    )
    scan = Scan(
        simulation,
        [Dimension("dose", values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")})],
    )
    observables = [
        Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml"),
        Formula("mid_rel", "mid / max(mid)"),
        PK("pk", "mid", dose="PODOSE_mid", route="oral"),
        Custom("t_above", time_above, "min", symbols=["mid"]),
    ]
    res = Simulator().run(
        MIDAZOLAM_SBML,
        scan,
        observables,
        keep=["mid", "pk.cmax", "pk.tmax", "pk.auc_inf_obs", "pk.thalf", "t_above"],
    )
    for name in ("pk.cmax", "pk.tmax", "pk.auc_inf_obs", "pk.thalf", "t_above"):
        print(name, res[name].values.round(3), res.units[name])
    return res


if __name__ == "__main__":
    run()
