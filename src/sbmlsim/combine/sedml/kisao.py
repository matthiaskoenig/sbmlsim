import logging
from collections import namedtuple

import libsedml


logger = logging.getLogger(__name__)

# ----------------
# KISAO MAPPINGS
# ----------------
KISAOS_CVODE = [
    "KISAO:0000019",  # CVODE
    "KISAO:0000433",  # CVODE-like method
    "KISAO:0000407",
    "KISAO:0000099",
    "KISAO:0000035",
    "KISAO:0000071",
    "KISAO:0000288",  # "BDF" cvode, stiff=true
    "KISAO:0000280",  # "Adams-Moulton" cvode, stiff=false
]
KISAOS_RK4 = [
    "KISAO:0000032",  # RK4 explicit fourth-order Runge-Kutta method
    "KISAO:0000064",  # Runge-Kutta based method
]
KISAOS_RK45 = [
    "KISAO:0000086",  # RKF45 embedded Runge-Kutta-Fehlberg 5(4) method
]
KISAOS_LSODA = [
    "KISAO:0000088",  # roadrunner doesn't have an lsoda solver so use cvode
]
KISAOS_GILLESPIE = [
    "KISAO:0000241",  # Gillespie-like method
    "KISAO:0000029",
    "KISAO:0000319",
    "KISAO:0000274",
    "KISAO:0000333",
    "KISAO:0000329",
    "KISAO:0000323",
    "KISAO:0000331",
    "KISAO:0000027",
    "KISAO:0000082",
    "KISAO:0000324",
    "KISAO:0000350",
    "KISAO:0000330",
    "KISAO:0000028",
    "KISAO:0000038",
    "KISAO:0000039",
    "KISAO:0000048",
    "KISAO:0000074",
    "KISAO:0000081",
    "KISAO:0000045",
    "KISAO:0000351",
    "KISAO:0000084",
    "KISAO:0000040",
    "KISAO:0000046",
    "KISAO:0000003",
    "KISAO:0000051",
    "KISAO:0000335",
    "KISAO:0000336",
    "KISAO:0000095",
    "KISAO:0000022",
    "KISAO:0000076",
    "KISAO:0000015",
    "KISAO:0000075",
    "KISAO:0000278",
]
KISAOS_NLEQ = [
    "KISAO:0000099",
    "KISAO:0000274",
    "KISAO:0000282",
    "KISAO:0000283",
    "KISAO:0000355",
    "KISAO:0000356",
    "KISAO:0000407",
    "KISAO:0000408",
    "KISAO:0000409",
    "KISAO:0000410",
    "KISAO:0000411",
    "KISAO:0000412",
    "KISAO:0000413",
    "KISAO:0000432",
    "KISAO:0000437",
]

# allowed algorithms for simulation type
KISAOS_STEADYSTATE = KISAOS_NLEQ
KISAOS_UNIFORMTIMECOURSE = (
    KISAOS_CVODE + KISAOS_RK4 + KISAOS_RK45 + KISAOS_GILLESPIE + KISAOS_LSODA
)
KISAOS_ONESTEP = KISAOS_UNIFORMTIMECOURSE

# supported algorithm parameters
KISAOS_ALGORITHMPARAMETERS = {
    "KISAO:0000209": ("relative_tolerance", float),  # the relative tolerance
    "KISAO:0000211": ("absolute_tolerance", float),  # the absolute tolerance
    "KISAO:0000220": ("maximum_bdf_order", int),  # the maximum BDF (stiff) order
    "KISAO:0000219": (
        "maximum_adams_order",
        int,
    ),  # the maximum Adams (non-stiff) order
    "KISAO:0000415": (
        "maximum_num_steps",
        int,
    ),  # the maximum number of steps that can be taken before exiting
    "KISAO:0000467": (
        "maximum_time_step",
        float,
    ),  # the maximum time step that can be taken
    "KISAO:0000485": (
        "minimum_time_step",
        float,
    ),  # the minimum time step that can be taken
    "KISAO:0000332": (
        "initial_time_step",
        float,
    ),  # the initial value of the time step for algorithms that change this value
    "KISAO:0000107": (
        "variable_step_size",
        bool,
    ),  # whether or not the algorithm proceeds with an adaptive step size or not
    "KISAO:0000486": (
        "maximum_iterations",
        int,
    ),  # [nleq] the maximum number of iterations the algorithm should take before exiting
    "KISAO:0000487": ("minimum_damping", float),  # [nleq] minimum damping value
    "KISAO:0000488": ("seed", int),  # the seed for stochastic runs of the algorithm
}


def is_supported_algorithm_for_simulation_type(kisao, simType):
    """Check Algorithm Kisao Id is supported for simulation.

    :return: is supported
    :rtype: bool
    """
    supported = []
    if simType == libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
        supported = KISAOS_UNIFORMTIMECOURSE
    elif simType == libsedml.SEDML_SIMULATION_ONESTEP:
        supported = KISAOS_ONESTEP
    elif simType == libsedml.SEDML_SIMULATION_STEADYSTATE:
        supported = KISAOS_STEADYSTATE
    return kisao in supported


def integrator_from_kisao(kisao):
    """RoadRunner integrator name for algorithm KisaoID.

    :param kisao: KisaoID
    :type kisao: str
    :return: RoadRunner integrator name.
    :rtype: str
    """
    if kisao in KISAOS_NLEQ:
        return "nleq2"
    if kisao in KISAOS_CVODE:
        return "cvode"
    if kisao in KISAOS_GILLESPIE:
        return "gillespie"
    if kisao in KISAOS_RK4:
        return "rk4"
    if kisao in KISAOS_RK45:
        return "rk45"
    if kisao in KISAOS_LSODA:
        logger.warn("Roadrunner does not support LSODA. Using CVODE instead.")
        return "cvode"  # just use cvode
    return None


def algorithm_parameter_to_parameter_key(par):
    """ Resolve the mapping between parameter keys and roadrunner integrator keys."""
    ParameterKey = namedtuple("ParameterKey", "key value dtype")
    kisao = par.getKisaoID()
    value = par.getValue()

    if kisao in KISAOS_ALGORITHMPARAMETERS:
        # algorithm parameter is in the list of parameters
        key, dtype = KISAOS_ALGORITHMPARAMETERS[kisao]
        if dtype is bool:
            # transform manually ! (otherwise all strings give True)
            if value == "true":
                value = True
            elif value == "false":
                value = False
        else:
            # cast to data type of parameter
            value = dtype(value)
        return ParameterKey(key, value, dtype)
    else:
        # algorithm parameter not supported
        logger.warning(f"Unsupported AlgorithmParameter: {kisao} = {value})")
        return None
