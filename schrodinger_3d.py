"""Solve the stationary 3D Schrodinger equation on a cubic grid.

Model units: hbar = mass = 1.
Equation: H psi = E psi, with H = -0.5 * Laplacian + V.
Boundary condition: psi = 0 on the six faces of the cube.
Potential formulas live in potentials.py. For a short complete usage example,
start with example.py.
"""

import argparse

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Re-export these names so existing scripts keep working after the module split.
from potentials import POTENTIAL_LABELS, make_potential


def make_grid(n, length):
    """Return n INTERIOR coordinates per axis and their spacing.

    The boundaries are at -length/2 and +length/2. Their known zero
    values are not included in the unknown wavefunction vector.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")

    spacing = length / (n + 1)
    coordinates = -length / 2 + spacing * np.arange(1, n + 1)
    return coordinates, spacing


def build_hamiltonian(potential, spacing):
    """Build the sparse Hamiltonian for a uniform cubic grid.

    Arrays and eigenvectors use NumPy's C order: z varies fastest.
    The same spacing is used in all three directions.
    Input potential shape: (n, n, n). Output sparse matrix shape: (n**3, n**3).
    """
    if np.iscomplexobj(potential):
        raise ValueError("potential must be real for this solver.")
    potential = np.asarray(potential, dtype=float)
    if potential.ndim != 3 or len(set(potential.shape)) != 1:
        raise ValueError("potential must have shape (n, n, n).")
    if potential.shape[0] < 2:
        raise ValueError("Each axis must have at least 2 interior points.")
    if not np.all(np.isfinite(potential)):
        raise ValueError("potential contains NaN or infinity.")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")

    n = potential.shape[0]

    # d2 psi / dx2 = (psi[i-1] - 2*psi[i] + psi[i+1]) / spacing**2.
    # Missing neighbors at the outer rows are the known zero boundaries.
    second_derivative = sparse.diags(
        [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    ) / spacing**2

    # Laplacian = d2/dx2 + d2/dy2 + d2/dz2.
    laplacian_xy = sparse.kronsum(
        second_derivative, second_derivative, format="csr"
    )
    laplacian_xyz = sparse.kronsum(
        laplacian_xy, second_derivative, format="csr"
    )

    kinetic_energy = -0.5 * laplacian_xyz
    potential_energy = sparse.diags(potential.ravel(order="C"), format="csr")
    return kinetic_energy + potential_energy


def solve_lowest_states(hamiltonian, number_of_states, spacing):
    """Return sorted energies and wavefunctions stored as columns.

    Normalize each state so sum(abs(psi)**2) * spacing**3 = 1.
    Eigenvalues are in the same energy units as the input potential.
    Output shapes: energies (number_of_states,), wavefunctions (n**3, number_of_states).
    """
    size = hamiltonian.shape[0]
    if not 1 <= number_of_states < size:
        raise ValueError("number_of_states must be between 1 and n**3 - 1.")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")

    # SA = smallest algebraic: select the lowest energies, including negatives.
    # A random starting vector avoids selecting only even-parity states.
    initial_vector = np.random.default_rng(0).normal(size=size)
    # Search beyond the requested cutoff so repeated energies have room to
    # converge before keeping the requested lowest states.
    candidate_count = min(size - 1, number_of_states + 8)
    energies, wavefunctions = eigsh(
        hamiltonian,
        k=candidate_count,
        which="SA",
        # A larger search space helps resolve repeated (degenerate) energies.
        ncv=min(size, max(40, 4 * candidate_count + 1)),
        tol=1e-10,
        v0=initial_vector,
    )

    order = np.argsort(energies)[:number_of_states]
    energies = energies[order]
    wavefunctions = wavefunctions[:, order]

    cell_volume = spacing**3
    norms = np.sqrt(np.sum(np.abs(wavefunctions)**2, axis=0) * cell_volume)
    wavefunctions = wavefunctions / norms
    return energies, wavefunctions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potential", choices=list(POTENTIAL_LABELS), default="harmonic")
    parser.add_argument("--n", type=int, default=24, help="Interior points per axis")
    parser.add_argument("--length", type=float, default=8.0, help="Cube side length")
    parser.add_argument("--states", type=int, default=6, help="Lowest states to calculate")
    args = parser.parse_args()
    if args.potential == "hydrogen" and args.n % 2:
        args.n += 1
        print(f"Hydrogen: using even n={args.n} to avoid sampling r=0.")

    # 1. Define the space and potential.
    coordinates, spacing = make_grid(args.n, args.length)
    potential = make_potential(coordinates, args.potential, length=args.length)

    # 2. Discretize H and solve H psi = E psi.
    hamiltonian = build_hamiltonian(potential, spacing)
    energies, wavefunctions = solve_lowest_states(hamiltonian, args.states, spacing)

    # 3. Check each computed eigenpair and its probability normalization.
    residuals = hamiltonian @ wavefunctions - wavefunctions * energies
    relative_residuals = (
        np.linalg.norm(residuals, axis=0)
        / np.linalg.norm(wavefunctions, axis=0)
        / np.maximum(1.0, np.abs(energies))
    )
    probabilities = np.sum(np.abs(wavefunctions)**2, axis=0) * spacing**3

    print("Units: hbar = mass = 1")
    print(f"Potential: {args.potential}; zero boundary at +/-{args.length / 2:g}")
    print(f"Interior grid: {args.n}^3; spacing: {spacing:.8g}")
    print("\nState         Energy        Probability    Relative residual")
    for state, energy in enumerate(energies):
        print(
            f"{state:5d}   {energy:14.9f}   {probabilities[state]:12.9f}"
            f"   {relative_residuals[state]:.3e}"
        )

    if args.potential == "box":
        exact_ground_energy = 3 * np.pi**2 / (2 * args.length**2)
        print(f"\nContinuum box ground energy: {exact_ground_energy:.9f}")
    elif args.potential == "harmonic":
        print("\nInfinite-domain harmonic ground energy: 1.5")
        print("This calculation also has finite-box and finite-spacing errors.")
    else:
        print(f"\n{POTENTIAL_LABELS[args.potential]}: no analytical reference supplied.")

    # Example: state 0 as a 3D array; its probability density is abs(psi)**2.
    # psi = wavefunctions[:, 0].reshape(potential.shape, order="C")


if __name__ == "__main__":
    main()
