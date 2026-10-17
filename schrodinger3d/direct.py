"""Solve the full Cartesian Hamiltonian, then optionally measure the results.

This module does not import hydrogen.py, request n/l/m, or select a reflection
sector. Its inputs are the grid, potential, and count. With basis="symmetry",
basis.py recombines vectors inside each degenerate level after the solve.
"""

import numpy as np

from .angular import angular_diagnostics
from .basis import potential_symmetries, symmetry_adapted_states, symmetry_expectations
from .potentials import make_potential
from .solver import make_grid, build_hamiltonian, solve_lowest_states


def solve_full_grid(n, length, potential_name, number_of_states, measure_angular=False,
                    basis="raw"):
    """Return coordinates, spacing, eigenpairs, and post-solve diagnostics.

    basis="raw" keeps the eigensolver's vectors. basis="symmetry" recombines
    them within each degenerate level using the potential's own symmetries.
    """
    if basis not in ("raw", "symmetry"):
        raise ValueError('basis must be "raw" or "symmetry".')
    if potential_name == "hydrogen" and n % 2:
        n += 1
    coordinates, spacing = make_grid(n, length)
    potential = make_potential(coordinates, potential_name, length=length)
    H = build_hamiltonian(potential, spacing)

    # No orbital/symmetry information enters this eigensolve.
    energies, states = solve_lowest_states(H, number_of_states, spacing)

    symmetries, basis_notes = [], []
    if basis == "symmetry":
        symmetries = potential_symmetries(potential)
        states, basis_notes = symmetry_adapted_states(states, potential.shape, energies,
                                                      symmetries)

    rows = []
    shell = max(1, n // 10)
    for index, (energy, state) in enumerate(zip(energies, states.T)):
        psi = state.reshape(potential.shape, order="C")
        probability = float(np.sum(abs(psi)**2) * spacing**3)
        interior = float(np.sum(abs(psi[shell:-shell, shell:-shell, shell:-shell])**2) * spacing**3)
        row = {
            "state": index, "energy": float(energy), "probability": probability,
            "relative_residual": float(np.linalg.norm(H @ state - energy * state)
                                       / np.linalg.norm(state) / max(1, abs(energy))),
            "outer_shell_probability": max(0.0, probability - interior),
        }
        if measure_angular:
            row.update(angular_diagnostics(psi, coordinates, spacing))
        if symmetries:
            row["symmetry_expectations"] = symmetry_expectations(psi, symmetries)
        rows.append(row)
    method = "Full Cartesian Hamiltonian eigensolve; post-solve measurements only"
    if basis == "symmetry":
        method = ("Full Cartesian Hamiltonian eigensolve; post-solve basis choice within "
                  "degenerate levels using the potential's symmetries")
    report = {
        "method": method,
        "potential": potential_name, "grid_points_per_axis": n,
        "length": length, "spacing": spacing, "unknowns": H.shape[0],
        "symmetry_reduction": False, "basis_rotation": basis == "symmetry",
        "basis": basis, "symmetries_used": symmetries, "basis_notes": basis_notes,
        "orbital_label_input": None, "angular_momentum_measured": measure_angular,
        "angular_discretization": "Centered first differences with zero exterior; boundary-sensitive",
        "states": rows,
    }
    return coordinates, spacing, energies, states, report
