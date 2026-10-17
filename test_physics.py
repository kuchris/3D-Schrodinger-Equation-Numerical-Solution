"""Independent physics checks for the discretization and eigensolver."""

import unittest
from unittest.mock import patch

import numpy as np

from schrodinger_3d import (
    build_hamiltonian,
    make_grid,
    make_potential,
    solve_lowest_states,
)


class PhysicsChecks(unittest.TestCase):
    def test_custom_harmonic_matches_builtin_spectrum(self):
        x, spacing = make_grid(8, 8)
        # Keep this test valid after the user edits their own function.
        with patch("custom_potential.V", side_effect=lambda X, Y, Z: 0.5 * (X**2 + Y**2 + Z**2)):
            custom = make_potential(x, "custom")
        builtin = make_potential(x, "harmonic")
        np.testing.assert_array_equal(custom, builtin)
        expected, _ = solve_lowest_states(build_hamiltonian(builtin, spacing), 1, spacing)
        actual, _ = solve_lowest_states(build_hamiltonian(custom, spacing), 1, spacing)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_custom_scalar_and_partial_coordinate_function(self):
        x, _ = make_grid(5, 8)
        with patch("custom_potential.V", return_value=-2.0):
            np.testing.assert_array_equal(make_potential(x, "custom"), np.full((5, 5, 5), -2.0))
        with patch("custom_potential.V", side_effect=lambda X, Y, Z: X + 2 * Y + 3 * Z):
            actual = make_potential(x, "custom")
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        np.testing.assert_array_equal(actual, X + 2 * Y + 3 * Z)

    def test_custom_rejects_invalid_potentials(self):
        x, _ = make_grid(5, 8)
        for value in [1j, np.nan, np.inf, "not a number", np.zeros((2, 3))]:
            with self.subTest(value=str(value)):
                with patch("custom_potential.V", return_value=value):
                    with self.assertRaises(ValueError):
                        make_potential(x, "custom")

    def test_operator_matches_pointwise_stencil_with_asymmetric_potential(self):
        # This catches axis/order errors hidden by symmetric example potentials.
        x, spacing = make_grid(5, 8)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        potential = X + 2 * Y + 3 * Z
        psi = np.random.default_rng(42).normal(size=potential.shape)
        padded = np.pad(psi, 1)
        laplacian = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
            + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
            - 6 * psi
        ) / spacing**2
        expected = -0.5 * laplacian + potential * psi
        actual = build_hamiltonian(potential, spacing) @ psi.ravel()
        np.testing.assert_allclose(actual.reshape(psi.shape), expected, atol=1e-13)

    def test_box_matches_discrete_spectrum_and_probability_integral(self):
        n, length = 10, 8
        x, spacing = make_grid(n, length)
        H = build_hamiltonian(make_potential(x, "box"), spacing)
        energies, states = solve_lowest_states(H, 4, spacing)
        modes = np.arange(1, n + 1)
        one_dimensional = (1 - np.cos(np.pi * modes / (n + 1))) / spacing**2
        exact_discrete = (
            one_dimensional[:, None, None]
            + one_dimensional[None, :, None]
            + one_dimensional[None, None, :]
        )
        np.testing.assert_allclose(energies, np.sort(exact_discrete.ravel())[:4], atol=1e-10)
        np.testing.assert_allclose(states.T @ states * spacing**3, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(H @ states, states * energies, atol=1e-9)

    def test_negative_energy_shift_preserves_lowest_states(self):
        x, spacing = make_grid(8, 8)
        V = make_potential(x, "harmonic")
        H = build_hamiltonian(V, spacing)
        shifted_H = build_hamiltonian(V - 20, spacing)
        energies, states = solve_lowest_states(H, 1, spacing)
        shifted_energies, shifted_states = solve_lowest_states(shifted_H, 1, spacing)
        np.testing.assert_allclose(shifted_energies, energies - 20, atol=1e-10)
        overlap = abs(states[:, 0] @ shifted_states[:, 0]) * spacing**3
        self.assertAlmostEqual(overlap, 1, places=10)

    def test_box_converges_to_continuum_at_second_order(self):
        exact = 3 * np.pi**2 / (2 * 8**2)
        errors = []
        spacings = []
        for n in [8, 16]:
            x, spacing = make_grid(n, 8)
            H = build_hamiltonian(make_potential(x, "box"), spacing)
            energies, _ = solve_lowest_states(H, 1, spacing)
            errors.append(abs(energies[0] - exact))
            spacings.append(spacing)
        order = np.log(errors[0] / errors[1]) / np.log(spacings[0] / spacings[1])
        self.assertGreater(order, 1.95)
        self.assertLess(order, 2.05)


if __name__ == "__main__":
    unittest.main()
