"""The sensitivity analyses of the toy model, a runner of the toy project.

The concentration of A decays with the rate `k` and does not depend on the
volume `Vc`. The group `slow` runs the variant `slow` (k = 0.1/min instead of
0.2/min), so its global designs vary `k` around 0.1/min.
"""

from sbmlsim.sensitivity.parameters import (
    SensitivityParameter,
    parameters_from_sbml,
    set_relative_bounds,
)
from sbmlsim.sensitivity.runner import (
    LocalAnalysis,
    SamplingAnalysis,
    SensitivityGroup,
    SensitivityOutput,
    SensitivityRunner,
)
from sbmlsim.simulation import PK, Formula, Observable, Simulation
from tests.project_toy import PROJECT

#: the outputs of the analyses; the third is named by another id than its
#: observable
OUTPUTS = [
    SensitivityOutput("A_auc", "A AUC∞"),
    SensitivityOutput("A_thalf", "A half-life"),
    SensitivityOutput("[A]_min", "A minimum", observable="A_min"),
]


def observables() -> list[Observable]:
    """Get the observables of the outputs, built when a run starts."""
    return [
        PK("pk_A", "[A]", parameters=["auc_inf_obs", "thalf"]),
        Formula("A_auc", "pk_A.auc_inf_obs", unit="mM*min"),
        Formula("A_thalf", "pk_A.thalf", unit="min"),
        Formula("A_min", "min([A])", unit="mM"),
    ]


def simulation(changes: dict[str, float]) -> Simulation:
    """Get the simulation of a group: the decay over an hour."""
    return Simulation(end=60, steps=600, time_unit="min", preinit_changes=changes)


#: the groups: the default variant and the slow decay
CONTROL = SensitivityGroup("control", "Control", simulation({}))
SLOW = SensitivityGroup("slow", "Slow decay", simulation({}), variant="slow")


def toy_parameters() -> list[SensitivityParameter]:
    """Get the parameters of the default variant, +-15 %."""
    parameters = parameters_from_sbml(PROJECT.model_path)
    set_relative_bounds(parameters, 0.15)
    return parameters


#: the runner of the toy project, `PROJECT.sensitivities["decay"]`
SENSITIVITY = SensitivityRunner(
    project=PROJECT,
    parameters=toy_parameters,
    outputs=OUTPUTS,
    observables=observables,
    analyses={
        "local": LocalAnalysis(groups=[CONTROL, SLOW]),
        "sampling": SamplingAnalysis(groups=[CONTROL, SLOW], n=8),
    },
    default=("local",),
    description="the local and sampled sensitivities of the decay",
)
