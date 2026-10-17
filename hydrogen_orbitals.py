"""Select real hydrogen orbitals using the same Cartesian finite differences.

Reflection symmetry reduces the calculation to one octant. Coordinate
permutations select s and real d orbitals from the computed eigenvectors.
No analytical orbital values are used to construct the wavefunction.
These cubic-box states approach the named free-atom orbitals on convergence.
"""

from itertools import permutations

import numpy as np
from scipy import sparse

from potentials import hydrogen_potential
from schrodinger_3d import make_grid, build_hamiltonian, solve_lowest_states


# label: (principal quantum number, reflection parity along x/y/z, selector, rank)
# A parity of -1 means the wavefunction changes sign under that reflection.
ORBITALS = {
    "1s": (1, (1, 1, 1), "s", 0),
    "2s": (2, (1, 1, 1), "s", 1),
    "3s": (3, (1, 1, 1), "s", 2),
    "2p_x": (2, (-1, 1, 1), "identity", 0),
    "2p_y": (2, (1, -1, 1), "identity", 0),
    "2p_z": (2, (1, 1, -1), "identity", 0),
    "3p_x": (3, (-1, 1, 1), "identity", 1),
    "3p_y": (3, (1, -1, 1), "identity", 1),
    "3p_z": (3, (1, 1, -1), "identity", 1),
    "3d_xy": (3, (-1, -1, 1), "identity", 0),
    "3d_xz": (3, (-1, 1, -1), "identity", 0),
    "3d_yz": (3, (1, -1, -1), "identity", 0),
    "3d_x2-y2": (3, (1, 1, 1), "d_difference", 0),
    "3d_z2": (3, (1, 1, 1), "d_z2", 0),
}


def octant_hamiltonian(coordinates, spacing, parity):
    """Restrict the original finite-difference H to a reflection sector.

    The first positive point is h/2. Its ghost neighbor at -h/2 equals
    parity * psi(h/2), adding -parity/(2*h**2) to the kinetic diagonal.
    This is an exact symmetry reduction of the full discrete matrix.
    """
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    potential = hydrogen_potential(x, y, z)
    H = build_hamiltonian(potential, spacing)
    correction = np.zeros(potential.shape)
    for axis, sign in enumerate(parity):
        face = [slice(None)] * 3
        face[axis] = 0
        correction[tuple(face)] -= sign / (2 * spacing**2)
    return H + sparse.diags(correction.ravel(), format="csr")


def select_symmetry(values, selector):
    """Project numerical values using coordinate permutations only."""
    if selector == "s":
        return sum(values.transpose(order) for order in permutations(range(3))) / 6
    if selector in ("d_difference", "d_z2"):
        difference = (values - values.swapaxes(0, 1)) / 2
        if selector == "d_z2":
            return difference.swapaxes(0, 2) - difference.swapaxes(1, 2)
        return difference
    return values


def extend_octant(values, parity):
    """Mirror a positive-octant state into the full cube, preserving norm."""
    full = values
    for axis, sign in enumerate(parity):
        full = np.concatenate([sign * np.flip(full, axis=axis), full], axis=axis)
    return full / np.sqrt(8)


def solve_hydrogen_orbital(n, length, label):
    """Return coordinates, spacing, energy, full real psi, and diagnostics.

    n is the FULL grid point count per axis, not the principal quantum number.
    Labels cover the low s/p/d states; they are not a general l,m classifier.
    A positive-energy finite-box state is rejected as a free-atom bound orbital.
    """
    if label not in ORBITALS:
        raise ValueError(f"Unknown hydrogen orbital: {label}")
    if n < 8:
        raise ValueError("Named hydrogen orbitals require at least 8 points per axis.")
    if n % 2:
        n += 1
    coordinates, spacing = make_grid(n, length)
    principal_n, parity, selector, rank = ORBITALS[label]
    positive_coordinates = coordinates[n // 2:]
    H = octant_hamiltonian(positive_coordinates, spacing, parity)
    count = 8 if selector != "identity" else rank + 1
    energies, states = solve_lowest_states(H, count, spacing)
    shape = (len(positive_coordinates),) * 3

    candidates = []
    for energy, state in zip(energies, states.T):
        projected = select_symmetry(state.reshape(shape), selector).ravel()
        if np.linalg.norm(projected) < 0.05 * np.linalg.norm(state):
            continue  # The state belongs to another symmetry representation.
        for _, previous in candidates:
            projected -= (previous @ projected * spacing**3) * previous
        norm = np.sqrt(projected @ projected * spacing**3)
        if norm < 1e-6:
            continue
        candidates.append((energy, projected / norm))
    if len(candidates) <= rank:
        raise RuntimeError("Orbital symmetry was not resolved; increase the grid or box.")
    energy, state = candidates[rank]
    if energy >= 0:
        raise ValueError(
            f"{label} is not bound in this finite box (E={energy:.6g}). "
            "Increase length while keeping the grid spacing sufficiently small."
        )
    residual = np.linalg.norm(H @ state - energy * state) / np.linalg.norm(state)
    if residual > 1e-7 * max(1, abs(energy)):
        raise RuntimeError("Selected symmetry combination has an excessive eigenpair residual.")
    psi = extend_octant(state.reshape(shape), parity)
    if psi.flat[np.argmax(abs(psi))] < 0:
        psi = -psi
    reference = -0.5 / principal_n**2
    density = psi**2
    shell = max(1, n // 10)
    interior_probability = density[shell:-shell, shell:-shell, shell:-shell].sum() * spacing**3
    info = {
        "label": label, "principal_n": principal_n,
        "reference_energy": reference,
        "relative_energy_error": float(abs(energy - reference) / abs(reference)),
        "relative_residual": float(residual / max(1, abs(energy))),
        "outer_shell_probability": float(1 - interior_probability),
        "method": "Cartesian finite differences with numerical symmetry selection",
    }
    return coordinates, spacing, float(energy), psi, info
