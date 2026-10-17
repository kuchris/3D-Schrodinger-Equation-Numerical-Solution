"""Post-solve basis choice inside degenerate levels, without orbital tables."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from schrodinger3d.basis import degenerate_levels, potential_symmetries
from schrodinger3d.direct import solve_full_grid
from schrodinger3d.potentials import make_potential
from schrodinger3d.solver import build_hamiltonian


def levels_with_signs(report, energies):
    """Yield (level indices, reflection signs per state) for degenerate levels."""
    for level in degenerate_levels(energies):
        if len(level) > 1:
            yield level, [tuple(round(report["states"][i]["symmetry_expectations"][name])
                                for name in ("Rx", "Ry", "Rz")) for i in level]


class SymmetryBasisChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # n <= 3 hydrogen shells: 1s, 2p, 2s, then nine n=3 states, 14 in all.
        cls.raw = solve_full_grid(32, 40, "hydrogen", 14, False, "raw")
        cls.sym = solve_full_grid(32, 40, "hydrogen", 14, False, "symmetry")

    def test_rotated_states_stay_eigenvectors_of_the_same_full_H(self):
        x, h, energies, states, report = self.sym
        H = build_hamiltonian(make_potential(x, "hydrogen", length=40), h)
        np.testing.assert_allclose(H @ states, states * energies, atol=1e-9)
        np.testing.assert_allclose(states.T @ states * h**3, np.eye(14), atol=1e-10)
        self.assertTrue(report["basis_rotation"])
        self.assertEqual(report["symmetries_used"], ["Rx", "Ry", "Rz", "Pxy"])
        self.assertEqual(report["basis_notes"], [])

    def test_each_level_keeps_the_raw_eigenspace(self):
        _, h, energies, raw_states, _ = self.raw
        _, _, sym_energies, sym_states, _ = self.sym
        np.testing.assert_array_equal(energies, sym_energies)
        for level in degenerate_levels(energies):
            raw, sym = raw_states[:, level], sym_states[:, level]
            # sym = raw @ coefficients exactly, with an orthogonal coefficient matrix.
            coefficients = raw.T @ sym * h**3
            np.testing.assert_allclose(coefficients.T @ coefficients, np.eye(len(level)),
                                       atol=1e-8)
            np.testing.assert_allclose(raw @ coefficients, sym, atol=1e-8)

    def test_triplets_resolve_into_distinct_reflection_signatures(self):
        _, _, energies, _, report = self.sym
        triplets = [signs for level, signs in levels_with_signs(report, energies)
                    if len(level) == 3]
        p_like = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        d_like = [(-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
        self.assertIn(p_like, triplets)
        self.assertIn(d_like, triplets)

    def test_even_pair_is_split_by_the_swap_and_matches_named_mode(self):
        from schrodinger3d.hydrogen import solve_hydrogen_orbital

        _, h, energies, states, report = self.sym
        pairs = [level for level, signs in levels_with_signs(report, energies)
                 if len(level) == 2 and signs == [(1, 1, 1)] * 2]
        self.assertEqual(len(pairs), 1)
        swap = [report["states"][i]["symmetry_expectations"]["Pxy"] for i in pairs[0]]
        np.testing.assert_allclose(swap, [-1, 1], atol=1e-8)
        # The named mode is used here only as an independent comparison.
        _, _, named_energy, named_z2, _ = solve_hydrogen_orbital(32, 40, "3d_z2")
        self.assertAlmostEqual(energies[pairs[0][1]], named_energy, places=9)
        overlap = abs(np.sum(states[:, pairs[0][1]] * named_z2.ravel()) * h**3)
        self.assertAlmostEqual(overlap, 1, places=8)

    def test_level_cut_by_state_count_is_left_unrotated_and_reported(self):
        _, _, _, raw_states, _ = solve_full_grid(32, 40, "hydrogen", 3, False, "raw")
        _, _, _, sym_states, report = solve_full_grid(32, 40, "hydrogen", 3, False, "symmetry")
        np.testing.assert_array_equal(sym_states, raw_states)
        self.assertEqual(len(report["basis_notes"]), 1)
        self.assertIn("left unrotated", report["basis_notes"][0])

    def test_symmetries_are_detected_from_the_potential_only(self):
        coordinates = np.linspace(-3, 3, 7)
        X, Y, Z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        self.assertEqual(potential_symmetries(X**2 + Y**2 + 2 * Z**2),
                         ["Rx", "Ry", "Rz", "Pxy"])
        self.assertEqual(potential_symmetries(X**2 + 2 * Y**2 + 3 * Z**2), ["Rx", "Ry", "Rz"])
        self.assertEqual(potential_symmetries(X**2 + Y**2 + Z**2 + 0.3 * Z), ["Rx", "Ry", "Pxy"])
        self.assertEqual(potential_symmetries(X + 2 * Y + 3 * Z), [])

    def test_potential_without_symmetry_is_left_unchanged(self):
        def potential(x, y, z):
            return 0.5 * (x*x + y*y + z*z) + 0.2*x + 0.3*y + 0.4*z
        with patch("custom.potential.V", side_effect=potential):
            _, _, _, raw_states, _ = solve_full_grid(8, 8, "custom", 4, False, "raw")
            _, _, _, sym_states, report = solve_full_grid(8, 8, "custom", 4, False, "symmetry")
        self.assertEqual(report["symmetries_used"], [])
        np.testing.assert_array_equal(sym_states, raw_states)

    def test_settings_accept_basis_only_in_full_mode(self):
        from schrodinger3d.plotting import parse_arguments

        self.assertEqual(parse_arguments(["--config", "config/direct.toml"]).basis, "symmetry")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--config", "config/hydrogen.toml", "--basis", "symmetry"])


if __name__ == "__main__":
    unittest.main()
