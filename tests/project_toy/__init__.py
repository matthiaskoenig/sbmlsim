"""A toy model package: the decay of a substance A in one compartment.

The tests of `sbmlsim.project` run the project of the package, `PROJECT`. The
model has two variants, the decay at the rate 0.2/min (`default`) and at
0.1/min (`slow`), which the factory (`factory.create_toy_models`) writes into
`models`. The experiments, the fits, the factory and the credibility
declaration are references, so importing the package imports none of them, nor
roadrunner. The scan `decay` scans the rate of the decay, the action `rates`
writes a table of the rates of the variants (`actions.write_rates`).
"""

from pathlib import Path

from sbmlsim.project import (
    CheckSettings,
    FigureOutputs,
    Project,
    ProjectAction,
    ScanAnalysis,
    SimulatorSettings,
)
from sbmlsim.release.settings import Author, ReleaseSettings
from sbmlsim.simulation import Dimension, Scan, Simulation

TOY_PATH = Path(__file__).parent


def decay_scan() -> Scan:
    """Get the scan of three rates of the decay [1/min]."""
    return Scan(
        Simulation(end=10, steps=20),
        [Dimension("rate", values={"k": [0.1, 0.2, 0.4]})],
    )


PROJECT = Project(
    name="toy",
    root=TOY_PATH,
    package=__name__,
    description="The decay of a substance A in one compartment.",
    variants={"default": Path("models/toy.xml"), "slow": Path("models/toy_slow.xml")},
    factory=".factory:create_toy_models",
    factory_outputs=[Path("models/toy.xml"), Path("models/toy_slow.xml")],
    experiments=".experiments:EXPERIMENTS",
    fits=".fitting:FIT_DEFINITIONS",
    fits_dir=Path("fits"),
    credibility=".credibility:CREDIBILITY",
    scans={
        "decay": ScanAnalysis(
            model=Path("models/toy.xml"),
            scan=decay_scan,
            description="three rates of the decay",
        )
    },
    actions={
        "rates": ProjectAction(
            function=".actions:write_rates",
            description="the table of the rates of the variants",
        ),
        "nothing": ".actions:nothing",
    },
    simulator=SimulatorSettings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, jit_optimization="O0"
    ),
    figures=FigureOutputs(formats=("svg", "png"), savefig_dpi=30),
    checks=CheckSettings(not_registered={"ToyUnregistered": "a test"}),
    release=ReleaseSettings(
        title="Toy model of the decay of a substance",
        repository="sbmlsim/toy-model",
        doi="10.5281/zenodo.0000000",
        authors=[Author("Doe, Jane", orcid="0000-0000-0000-0000", affiliation="Lab")],
        keywords=["toy", "decay"],
        docker_image="sbmlsim/toy",
        funding="No funding.",
        description="The decay of a substance A in one compartment.",
        first_year=2026,
    ),
)
"""The project of the toy model."""
