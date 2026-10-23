"""Uncertainty of the pharmacokinetics of midazolam with and without an inhibitor.

The dose and the body weight are uncertain inputs of the simulation, the
hepatic clearance is an uncertain parameter, the body weight and the height
are correlated. The propagation runs them through a control and an inhibitor
scenario with the same draws, summarizes the AUC and the Cmax with their
Monte Carlo errors, ranks the contributions of the uncertain values and
states the probability of a relevant interaction.
"""

from pathlib import Path

from sbmlsim import Q
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import PK, Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import (
    Covariance,
    TestReport,
    Uncertain,
    UncertaintyModel,
    propagate,
)

#: the dose, the body weight and the height of the group, as reported
DOSE = Uncertain(Q(5, "mg"), cv=0.05)
WEIGHT = Uncertain(Q(75, "kg"), sd=Q(12, "kg"), low=Q(45, "kg"), kind="variability")
HEIGHT = Uncertain(Q(176, "cm"), sd=Q(8, "cm"), kind="variability")


def simulation(activity: float = 1.0) -> Simulation:
    """Get the simulation of an oral dose, with the hepatic CYP3A4 activity."""
    preinit: dict = {"BW": WEIGHT, "HEIGHT": HEIGHT, "LI__MIDOH_f": activity}
    return Simulation(
        end=24,
        time_unit="hr",
        steps=96,
        preinit_changes=preinit,
        changes=[Change(0, {"PODOSE_mid": DOSE})],
    )


def run(output: Path) -> None:
    """Run the analysis and write its tables."""
    uq = UncertaintyModel(
        # a parameter from the literature with a CV of 30 %
        parameters={"LI__MIDOH_Vmax": Uncertain(0.005, cv=0.3)},
        covariance=[Covariance.correlation({("BW", "HEIGHT"): 0.6})],
    )
    res = propagate(
        MIDAZOLAM_SBML,
        # an inhibitor of CYP3A4 leaves 30 % of its activity
        {"control": simulation(), "inhibitor": simulation(activity=0.3)},
        uq,
        [PK("pk", "[Cve_mid]", dose="PODOSE_mid", route="oral")],
        outputs=["pk.auc_last", "pk.cmax"],
        n_start=256,
        n_max=8192,
        seed=1234,
        simulator=Simulator(n_workers=1),
    )
    summary = res.summary(["pk.auc_last", "pk.cmax"])
    print(
        summary[["scenario", "output", "n", "mean", "mean_se", "sd", "q0.05", "q0.95"]]
    )
    print(res.contributions("pk.auc_last", scenario="control"))
    ratio = res.contrast("pk.auc_last", "inhibitor", "control")
    report = TestReport(
        [
            res.probability(ratio, greater=1.25, at_least=0.9),
            res.probability(ratio, greater=3.0),
        ]
    )
    table = report.table()
    print(table[["hypothesis", "estimate", "low", "high", "decision"]])
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "uncertainty_summary.tsv", sep="\t", index=False)
    table.to_csv(output / "uncertainty_tests.tsv", sep="\t", index=False)


if __name__ == "__main__":
    run(Path("results") / "uncertainty_midazolam")
