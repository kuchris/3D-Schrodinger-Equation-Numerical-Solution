"""Independent physics checks for the discretization and eigensolver."""

import unittest
from unittest.mock import patch

import numpy as np

from schrodinger3d.potentials import POTENTIAL_LABELS, make_potential
from schrodinger3d.solver import (
    build_hamiltonian,
    make_grid,
    solve_lowest_states,
)


class PhysicsChecks(unittest.TestCase):
    def test_paper_presets_solve_with_normalized_eigenpairs(self):
        x, spacing = make_grid(8, 8)
        for kind in POTENTIAL_LABELS:
            if kind == "custom":
                continue  # The user owns this function; tested separately below.
            with self.subTest(potential=kind):
                V = make_potential(x, kind, length=8)
                self.assertEqual(V.shape, (8, 8, 8))
                self.assertTrue(np.all(np.isfinite(V)))
                H = build_hamiltonian(V, spacing)
                energies, states = solve_lowest_states(H, 2, spacing)
                np.testing.assert_allclose(H @ states, states * energies, atol=1e-8)
                np.testing.assert_allclose(states.T @ states * spacing**3, np.eye(2), atol=1e-9)

    def test_paper_harmonic_matches_independent_separable_solution(self):
        from scipy.linalg import eigh_tridiagonal

        x, spacing = make_grid(9, 8)
        one_axis_energy = eigh_tridiagonal(
            1 / spacing**2 + 0.00005 * x**2,
            -0.5 * np.ones(8) / spacing**2,
            select="i", select_range=(0, 0),
        )[0][0]
        H = build_hamiltonian(make_potential(x, "paper_harmonic"), spacing)
        energies, _ = solve_lowest_states(H, 1, spacing)
        self.assertAlmostEqual(energies[0], 3 * one_axis_energy, places=10)

    def test_lowest_box_states_include_first_excited_triplet(self):
        # Small Krylov searches previously skipped one member at these grids.
        for n in (12, 17):
            with self.subTest(n=n):
                x, spacing = make_grid(n, 8)
                H = build_hamiltonian(make_potential(x, "box"), spacing)
                energies, _ = solve_lowest_states(H, 4, spacing)
                e1 = (1 - np.cos(np.pi / (n + 1))) / spacing**2
                e2 = (1 - np.cos(2 * np.pi / (n + 1))) / spacing**2
                np.testing.assert_allclose(energies, [3 * e1] + [2 * e1 + e2] * 3,
                                           atol=1e-10)

    def test_circular_keeps_appendix_a_step_and_axial_behavior(self):
        x, _ = make_grid(9, 8)
        V = make_potential(x, "circular", length=8)
        self.assertEqual(set(np.unique(V)), {0, 400})
        self.assertEqual(V[0, 0, 4], 400)
        np.testing.assert_array_equal(V[4, 4, :], np.zeros(9))

    def test_torus_uses_section_iv_ring_radius_and_energy_offset(self):
        # At rho=0.4L and z=0, Eq. (31) has minimum -(r/L)**2=-0.01.
        x = np.array([-3.2, 0, 3.2])
        V = make_potential(x, "torus", length=8)
        self.assertAlmostEqual(V[0, 1, 1], -0.01, places=14)
        self.assertAlmostEqual(V[1, 2, 1], -0.01, places=14)
        self.assertGreater(V[1, 1, 1], V[0, 1, 1])

    def test_tanh_retains_z_independence(self):
        x, _ = make_grid(9, 8)
        V = make_potential(x, "tanh", length=8)
        np.testing.assert_array_equal(V[:, :, 0], V[:, :, 4])
        np.testing.assert_array_equal(V[:, :, 0], V[:, :, -1])
        self.assertEqual(V[4, 4, 4], 0)
        self.assertGreater(V[0, 0, 4], 0)

    def test_hydrogen_preserves_coulomb_without_softening(self):
        V = make_potential(np.array([-1., 1.]), "hydrogen", length=6)
        np.testing.assert_allclose(V, -np.ones((2, 2, 2)) / np.sqrt(3))
        x, _ = make_grid(9, 8)
        with self.assertRaisesRegex(ValueError, "singular at the origin"):
            make_potential(x, "hydrogen", length=8)

    def test_paper_options_reach_plotter_without_false_reference(self):
        from schrodinger3d.plotting import parse_arguments, reference_energies

        for kind in ("paper_harmonic", "circular", "torus", "tanh", "hydrogen"):
            with self.subTest(potential=kind):
                args = parse_arguments(["--potential", kind, "--n", "25"])
                self.assertEqual(args.potential, kind)
                self.assertEqual(args.n, 26 if kind == "hydrogen" else 25)
                self.assertIsNone(reference_energies(kind, 8, 6))

    def test_custom_harmonic_matches_builtin_spectrum(self):
        x, spacing = make_grid(8, 8)
        # Keep this test valid after the user edits their own function.
        with patch("custom.potential.V", side_effect=lambda X, Y, Z: 0.5 * (X**2 + Y**2 + Z**2)):
            custom = make_potential(x, "custom")
        builtin = make_potential(x, "harmonic")
        np.testing.assert_array_equal(custom, builtin)
        expected, _ = solve_lowest_states(build_hamiltonian(builtin, spacing), 1, spacing)
        actual, _ = solve_lowest_states(build_hamiltonian(custom, spacing), 1, spacing)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_custom_scalar_and_partial_coordinate_function(self):
        x, _ = make_grid(5, 8)
        with patch("custom.potential.V", return_value=-2.0):
            np.testing.assert_array_equal(make_potential(x, "custom"), np.full((5, 5, 5), -2.0))
        with patch("custom.potential.V", side_effect=lambda X, Y, Z: X + 2 * Y + 3 * Z):
            actual = make_potential(x, "custom")
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        np.testing.assert_array_equal(actual, X + 2 * Y + 3 * Z)

    def test_custom_rejects_invalid_potentials(self):
        x, _ = make_grid(5, 8)
        for value in [1j, np.nan, np.inf, "not a number", np.zeros((2, 3))]:
            with self.subTest(value=str(value)):
                with patch("custom.potential.V", return_value=value):
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
