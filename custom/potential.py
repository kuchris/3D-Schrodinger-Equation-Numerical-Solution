"""Edit V below, save, then double-click plot.cmd.

Select potential = "custom" in config/settings.toml.
Units: hbar = mass = 1. Do not multiply V by the grid spacing.
The cube still has zero-wavefunction boundary conditions on all six faces.
"""

import numpy as np


def V(x, y, z):
    """Return a real, finite potential evaluated on NumPy coordinate arrays.

    Use NumPy operations such as np.exp, np.sin, np.sqrt, and np.where.
    A scalar return value also works for a constant potential.
    """
    # Example: harmonic oscillator. Replace this expression with your function.
    return 0.5 * (x**2 + y**2 + z**2)

    # Other examples (replace the return above):
    # Double well along x, harmonic confinement along y and z:
    # return 0.2 * (x**2 - 2.0**2)**2 + 0.5 * (y**2 + z**2)

    # Attractive Gaussian well:
    # return -5.0 * np.exp(-(x**2 + y**2 + z**2) / (2 * 1.0**2))

    # Finite spherical well (depth 5, radius 2):
    # return np.where(x**2 + y**2 + z**2 < 2.0**2, -5.0, 0.0)
