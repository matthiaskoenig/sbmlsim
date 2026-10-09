"""Models with known sensitivities."""

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
