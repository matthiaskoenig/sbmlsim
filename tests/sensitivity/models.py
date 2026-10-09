"""Models with known sensitivities."""

from typing import Any

import numpy as np

#: y = a^2 * b^-1 * c^0.5: the normalized local sensitivities are 2, -1 and 0.5
POWER_LAW = """
model powerlaw
  a = 2; b = 3; c = 4
  y := a^2 * b^(-1) * c^0.5
end
"""

#: the Ishigami function of x1, x2, x3 uniform in [-pi, pi], a = 7, b = 0.1:
#: S1 = (0.314, 0.442, 0), ST = (0.558, 0.442, 0.244)
ISHIGAMI = """
model ishigami
  x1 = 0; x2 = 0; x3 = 0
  y := sin(x1) + 7 * sin(x2)^2 + 0.1 * x3^4 * sin(x1)
end
"""


#: the chain S1 -> S2 -> S3 of the sensitivity example with the rates k1 and k2:
#: S1 + S2 + S3 is conserved and S3 saturates at the initial S1
CHAIN = """
model chain
  compartment V = 1
  species S1 in V = 1; species S2 in V = 0; species S3 in V = 0
  k1 = 1; k2 = 1
  R1: S1 -> S2; k1*S1
  R2: S2 -> S3; k2*S2
end
"""


def s2_end_fails_for_high_s1_and_k1(time: np.ndarray, values: dict[str, Any]) -> float:
    """The last value of `[S2]`; it fails for an initial `[S1]` above 5 and `k1` above 1."""
    if values["[S1]"][0] > 5.0 and values["k1"][0] > 1.0:
        raise ValueError("k1 is too large for a high [S1]")
    return float(values["[S2]"][-1])
