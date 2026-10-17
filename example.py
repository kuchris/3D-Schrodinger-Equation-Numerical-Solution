"""Start here to read the complete physics workflow. Run: python example.py."""

import numpy as np

from potentials import make_potential
from schrodinger_3d import make_grid, build_hamiltonian, solve_lowest_states


def main():
    # 1. Choose a model and a spatial grid.
    grid_points_per_axis = 25
    box_length = 8.0
    potential_name = "harmonic"
    number_of_states = 4

    coordinates, spacing = make_grid(grid_points_per_axis, box_length)

    # 2. Evaluate V and build H = -0.5 * Laplacian + V.
    potential = make_potential(coordinates, potential_name, length=box_length)
    hamiltonian = build_hamiltonian(potential, spacing)

    # 3. Solve H psi = E psi for the lowest states.
    energies, wavefunctions = solve_lowest_states(hamiltonian, number_of_states, spacing)

    # 4. Reshape the ground state and check its total probability.
    ground_state = wavefunctions[:, 0].reshape(potential.shape, order="C")
    probability_density = np.abs(ground_state)**2
    total_probability = np.sum(probability_density) * spacing**3

    print("Energies:", energies)
    print(f"Ground-state probability integral: {total_probability:.12f}")


if __name__ == "__main__":
    main()
