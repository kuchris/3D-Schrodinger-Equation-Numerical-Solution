"""Potential formulas, evaluated in model units with hbar = mass = 1.

Read the named functions below to inspect a formula. The application calls
make_potential() to select one and evaluate it on a cubic coordinate grid.
Paper sources and unit caveats: docs/paper-potentials.md.
"""

import numpy as np
from scipy.signal import square


POTENTIAL_LABELS = {
    "box": "Particle in a 3D box",
    "harmonic": "3D harmonic oscillator",
    "paper_harmonic": "Paper harmonic potential (Eq. 29)",
    "circular": "Circular well (paper Appendix A)",
    "torus": "Torus potential (Eq. 31)",
    "tanh": "Tanh potential (Eq. 32)",
    "hydrogen": "Hydrogen-like Coulomb potential (Eq. 33)",
    "custom": "Custom potential V(x, y, z)",
}


def box_potential(x, y, z):
    """Zero interior potential. The solver imposes the infinite cube walls."""
    return np.zeros_like(x)


def harmonic_potential(x, y, z, omega=1.0):
    """V = omega**2 * r**2 / 2, with mass = 1.

    omega=1 is the analytical benchmark; omega=0.01 is paper Eq. (29).
    """
    radius_squared = x**2 + y**2 + z**2
    return omega**2 * radius_squared / 2


def circular_potential(x, y, z, length, barrier_height=400.0, radial_frequency=2.1):
    """Paper Appendix A's SP expression; values are 0 or barrier_height.

    square() is the periodic square wave. The z term is outside sqrt.
    This preserves the printed code's geometry, not a spherical step well.
    """
    scaled_radius_xy = np.sqrt((x / length)**2 + (y / length)**2)
    phase = radial_frequency * np.pi * scaled_radius_xy + (z / length)**2
    return (barrier_height / 2) * (1 - square(phase))


def torus_potential(x, y, z, length, major_radius_fraction=0.4, minor_radius_fraction=0.1):
    """Paper Eq. (31), with R/L=0.4 and r/L=0.1 from Section IV.D.

    The minor-radius term is a constant energy shift; it does not control
    the width of the eigenfunction.
    """
    scaled_radius_xy = np.sqrt((x / length)**2 + (y / length)**2)
    radial_potential = (scaled_radius_xy - major_radius_fraction)**2
    axial_potential = (z / length)**2
    energy_offset = -minor_radius_fraction**2
    return radial_potential + axial_potential + energy_offset


def tanh_potential(x, y, z, length):
    """Paper Eq. (32): tanh((x/L)**2 + (y/L)**2), independent of z."""
    return np.tanh((x / length)**2 + (y / length)**2)


def hydrogen_potential(x, y, z, origin_tolerance=0.0):
    """Paper Eq. (33): exact point-sampled -1/r, without softening."""
    radius = np.sqrt(x**2 + y**2 + z**2)
    if np.any(radius <= origin_tolerance):
        raise ValueError(
            "The Coulomb potential is singular at the origin. "
            "Use an even n (the plotting app adjusts odd n automatically)."
        )
    return -1 / radius


def make_potential(coordinates, kind, length=None):
    """Evaluate a selected potential on a uniform cubic grid.

    coordinates: 1D array of interior positions shared by the three axes.
    kind: a name listed in POTENTIAL_LABELS.
    length: cube side length; inferred from the uniform grid when omitted.

    Returns a real array with shape (n, n, n), where V[i,j,k] is the
    potential at (coordinates[i], coordinates[j], coordinates[k]).
    """
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    if length is None:
        spacing = coordinates[1] - coordinates[0]
        length = spacing * (len(coordinates) + 1)

    if kind == "box":
        return box_potential(x, y, z)
    if kind == "harmonic":
        return harmonic_potential(x, y, z, omega=1.0)
    if kind == "paper_harmonic":
        return harmonic_potential(x, y, z, omega=0.01)
    if kind == "circular":
        return circular_potential(x, y, z, length)
    if kind == "torus":
        return torus_potential(x, y, z, length)
    if kind == "tanh":
        return tanh_potential(x, y, z, length)
    if kind == "hydrogen":
        spacing = length / (len(coordinates) + 1)
        return hydrogen_potential(x, y, z, origin_tolerance=spacing * 1e-10)
    if kind == "custom":
        from custom.potential import V

        values = np.asarray(V(x, y, z))
        if np.iscomplexobj(values):
            raise ValueError("custom.potential.V must return real values.")
        try:
            values = np.broadcast_to(values, x.shape).astype(float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "custom.potential.V must return numeric values broadcastable "
                f"to the grid shape {x.shape}; got shape {values.shape}."
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError("custom.potential.V contains NaN or infinity.")
        return values

    raise ValueError(f"Unknown potential: {kind}")
