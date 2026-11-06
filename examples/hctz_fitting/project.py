"""The project of the HCTZ model: its experiments, fits, scans and uncertainties.

`PROJECT` is what the run layer resolves, e.g.

    python -m sbmlsim.run list examples.hctz_fitting.project:PROJECT

Every callable builds a deterministic object: the designs and the draws have
fixed seeds, so every unit of a run builds the same scan. The analyses are
small, so that the tests run them.
"""

import numpy as np

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH, MODEL_PATH
from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from examples.hctz_fitting.simulations import EXPERIMENTS
from sbmlsim import Q
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.project import (
    ADOPTED_FILE,
    Project,
    ScanAnalysis,
    UncertaintyAnalysis,
)
from sbmlsim.simulation import (
    PK,
    Change,
    Dimension,
    Formula,
    Observable,
    Scan,
    Simulation,
)
from sbmlsim.simulation.sampling import LogNormal, Uniform, sobol
from sbmlsim.uncertainty import UncertaintyModel

#: the time grid of the dose scan in the time unit of the model (min), every
#: 30 minutes over 24 hours
DOSE_TIMES: tuple[float, ...] = tuple(float(t) for t in np.linspace(0, 24 * 60, 49))


def experiments() -> list[type[SimulationExperiment]]:
    """Get the simulation experiments of the studies."""
    return EXPERIMENTS["all"]


def observables() -> list[Observable]:
    """Get the plasma concentration in ng/ml and its pharmacokinetics."""
    return [
        Formula("cp", "[Cve_hctz] * Mr_hctz", unit="ng/ml"),
        PK("pk", "cp", dose="PODOSE_hctz", route="oral"),
    ]


def oral_dose() -> Simulation:
    """Get an oral dose of 25 mg over one day."""
    return Simulation(
        time_unit="hr",
        end=24,
        changes=[Change(0, {"PODOSE_hctz": Q(25, "mg")})],
    )


def dose_scan() -> Scan:
    """Get the scan of three oral doses."""
    return Scan(
        oral_dose(),
        [
            Dimension(
                "dose",
                values={"PODOSE_hctz": Q([12.5, 25, 50], "mg")},
                labels=["12.5 mg", "25 mg", "50 mg"],
            )
        ],
    )


def sobol_scan() -> Scan:
    """Get the Sobol design of the dissolution and the renal excretion."""
    simulation = oral_dose()
    design = sobol(
        {
            "Ka_dis_hctz": Uniform(relative=0.5),
            "KI__HCTZEX_k": Uniform(relative=0.5),
        },
        8,
        seed=1,
        model=MODEL_PATH,
        simulation=simulation,
    )
    return Scan(simulation, [design])


def renal_clearance() -> UncertaintyModel:
    """Get the uncertainty of the renal excretion, 30 %."""
    return UncertaintyModel(parameters={"KI__HCTZEX_k": LogNormal(cv=0.3)})


PROJECT = Project(
    name="hctz",
    root=HCTZ_PATH,
    experiments=experiments,
    base_path=HCTZ_PATH,
    data_path=DATA_PATH,
    fits=FIT_DEFINITIONS,
    adopted=HCTZ_PATH / ADOPTED_FILE,
    scans={
        "dose": ScanAnalysis(
            model=MODEL_PATH,
            scan=dose_scan,
            observables=observables,
            time=DOSE_TIMES,
            description="oral doses of 12.5, 25 and 50 mg over one day",
        ),
        "sobol": ScanAnalysis(
            model=MODEL_PATH,
            scan=sobol_scan,
            observables=observables,
            keep=["pk.cmax", "pk.auc_inf_obs"],
            indices="sobol",
            description=(
                "Sobol design of the dissolution and the renal excretion of "
                "a 25 mg oral dose"
            ),
        ),
    },
    uncertainties={
        "cl": UncertaintyAnalysis(
            model=MODEL_PATH,
            simulation=oral_dose,
            uq=renal_clearance,
            observables=observables,
            outputs=["pk.auc_inf_obs"],
            n=32,
            seed=1,
            description=(
                "the AUC of a 25 mg oral dose under a renal excretion uncertain by 30 %"
            ),
        ),
    },
    description="Hydrochlorothiazide (HCTZ) physiologically based pharmacokinetics",
)
