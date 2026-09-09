"""Parameters to optimize."""

from sbmlsim.fit import FitParameter

parameters_pk: list[FitParameter] = [
    # absorption
    FitParameter(
        pid="Ka_dis_hctz",
        start_value=1.0,
        lower_bound=1e-4,
        upper_bound=10,
        unit="1/hr",
    ),
    FitParameter(
        pid="GU__HCTZABS_k",
        lower_bound=1e-4,
        start_value=0.02,
        upper_bound=10,
        unit="1/min",
    ),
    FitParameter(
        pid="KI__HCTZEX_k",
        lower_bound=1e-10,
        start_value=1e-6,
        upper_bound=1,
        unit="1/ml",
    ),
]
