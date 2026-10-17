"""Measure angular momentum AFTER a Cartesian eigenproblem is solved.

Only coordinate multiplications and finite differences are used. There are
no spherical harmonics, orbital templates, reflection rules, or state rotations.
Finite cubes/grids do not have exact continuum rotational symmetry.
"""

import numpy as np


def first_derivative(values, spacing, axis):
    """Centered derivative with the same zero exterior values as the grid."""
    padding = [(0, 0)] * 3
    padding[axis] = (1, 1)
    padded = np.pad(values, padding)
    forward = [slice(None)] * 3
    backward = [slice(None)] * 3
    forward[axis] = slice(2, None)
    backward[axis] = slice(None, -2)
    return (padded[tuple(forward)] - padded[tuple(backward)]) / (2 * spacing)


def apply_angular_component(psi, coordinates, spacing, axis):
    """Apply L_axis = -i * (r_j * d_k - r_k * d_j), with hbar=1."""
    j, k = ((1, 2), (2, 0), (0, 1))[axis]
    shape_j = [1, 1, 1]
    shape_k = [1, 1, 1]
    shape_j[j] = len(coordinates)
    shape_k[k] = len(coordinates)
    r_j = coordinates.reshape(shape_j)
    r_k = coordinates.reshape(shape_k)
    return -1j * (r_j * first_derivative(psi, spacing, k)
                  - r_k * first_derivative(psi, spacing, j))


def angular_diagnostics(psi, coordinates, spacing):
    """Return moments, spreads, and a continuous l estimate, without editing psi.

    l_effective solves l*(l+1)=<L2>; it is not a certified quantum-number label.
    In particular, <Lz>=0 with a nonzero spread does NOT establish m=0.
    """
    norm_squared = float(np.vdot(psi, psi).real)
    if norm_squared <= 0:
        raise ValueError("Cannot measure angular momentum of a zero state.")
    components = [apply_angular_component(psi, coordinates, spacing, axis) for axis in range(3)]
    L2_psi = sum(apply_angular_component(component, coordinates, spacing, axis)
                 for axis, component in enumerate(components))
    L2_mean = float(np.vdot(psi, L2_psi).real / norm_squared)
    L2_std = float(np.linalg.norm(L2_psi - L2_mean * psi) / np.sqrt(norm_squared))
    Lz_mean = float(np.vdot(psi, components[2]).real / norm_squared)
    Lz_std = float(np.linalg.norm(components[2] - Lz_mean * psi) / np.sqrt(norm_squared))
    reflections = [float(np.vdot(psi, np.flip(psi, axis=axis)).real / norm_squared)
                   for axis in range(3)]
    return {
        "L2_mean": L2_mean, "L2_std": L2_std,
        "l_effective": float((np.sqrt(max(0, 1 + 4 * L2_mean)) - 1) / 2),
        "Lz_mean": Lz_mean, "Lz_std": Lz_std,
        "reflection_expectations_xyz": reflections,
    }
