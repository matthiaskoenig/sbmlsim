"""A toy model package with every kind of evidence of a credibility declaration.

It simulates the toy model of `tests.project_toy` (the decay of A) in two
experiments whose mappings are calibration, dose, population and limitation
evidence, unclaimed validation data and an outlier. The metadata of the
mappings is a generic dataclass of the package (`metadata.ToyMetaData`, the
route and the health of a group); the renal impairment is a label of the
metadata only, the model does not change. The declaration, the experiments,
the fits, the extension and the battery hook are references, so importing the package imports
none of them.
"""

from pathlib import Path

from sbmlsim.project import CredibilitySettings, Project, SimulatorSettings

CREDIBILITY_PATH = Path(__file__).parent
#: the SBML of the toy project, which this package simulates
MODEL_PATH = CREDIBILITY_PATH.parent / "project_toy" / "models" / "toy.xml"

PROJECT = Project(
    name="credibility",
    root=CREDIBILITY_PATH,
    package=__name__,
    title="Toy model with every kind of evidence",
    variants={"default": MODEL_PATH},
    experiments=".experiments:EXPERIMENTS",
    fits=".fitting:FIT_DEFINITIONS",
    fits_dir=Path("fits"),
    credibility=".credibility:CREDIBILITY",
    credibility_settings=CredibilitySettings(
        metadata_type=".metadata:ToyMetaData",
        domain_fields=("route", "health"),
        battery=".battery:BATTERY",
    ),
    simulator=SimulatorSettings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, jit_optimization="O0"
    ),
)
