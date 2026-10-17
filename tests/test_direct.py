"""Full-Hamiltonian solutions and diagnostics without orbital constraints."""

import unittest
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.linalg import eigh

from schrodinger3d.direct import solve_full_grid
from schrodinger3d.angular import angular_diagnostics
from schrodinger3d.solver import make_grid, build_hamiltonian


class DirectChecks(unittest.TestCase):
    def test_full_ui_route_does_not_import_orbital_rules(self):
        script = """
import sys
from schrodinger3d.plotting import parse_arguments
from schrodinger3d.direct import solve_full_grid
parse_arguments(['--config', 'config/direct.toml'])
solve_full_grid(8, 12, 'hydrogen', 4, True)
assert 'schrodinger3d.hydrogen' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_post_analysis_preserves_original_eigenpairs(self):
        _, _, energies, states, _ = solve_full_grid(8, 12, "hydrogen", 4, False)
        _, _, measured_energies, measured_states, _ = solve_full_grid(8, 12, "hydrogen", 4, True)
        np.testing.assert_array_equal(energies, measured_energies)
        np.testing.assert_array_equal(states, measured_states)

    def test_full_grid_does_not_call_named_orbital_solver(self):
        with patch("schrodinger3d.hydrogen.solve_hydrogen_orbital", side_effect=AssertionError):
            x, h, energies, states, report = solve_full_grid(10, 12, "hydrogen", 4, True)
        self.assertEqual(states.shape, (10**3, 4))
        self.assertEqual(report["unknowns"], 10**3)
        self.assertFalse(report["symmetry_reduction"])
        self.assertFalse(report["basis_rotation"])

    def test_asymmetric_potential_matches_independent_dense_eigensolve(self):
        def potential(x, y, z):
            return 0.5 * (x*x + y*y + z*z) + 0.2*x + 0.3*x*y
        with patch("custom.potential.V", side_effect=potential):
            x, h, energies, states, report = solve_full_grid(4, 8, "custom", 4, True)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        H = build_hamiltonian(potential(X, Y, Z), h)
        expected = eigh(H.toarray(), eigvals_only=True)[:4]
        np.testing.assert_allclose(energies, expected, atol=1e-10)
        np.testing.assert_allclose(H @ states, states * energies, atol=1e-9)
        np.testing.assert_allclose(states.T @ states * h**3, np.eye(4), atol=1e-10)

    def test_angular_measurement_does_not_modify_wavefunction(self):
        x, h = make_grid(25, 8)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        psi = Z * np.exp(-0.5 * (X*X + Y*Y + Z*Z))
        original = psi.copy()
        info = angular_diagnostics(psi, x, h)
        np.testing.assert_array_equal(psi, original)
        self.assertLess(abs(info["L2_mean"] - 2), 0.15)
        self.assertLess(info["Lz_std"], 0.05)

    def test_nonzero_Lz_spread_is_not_reported_as_definite_m_zero(self):
        x, h = make_grid(25, 8)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        psi = X * np.exp(-0.5 * (X*X + Y*Y + Z*Z))
        info = angular_diagnostics(psi, x, h)
        self.assertAlmostEqual(info["Lz_mean"], 0, places=12)
        self.assertGreater(info["Lz_std"], 0.8)
        self.assertNotIn("m", info)

    def test_complex_state_has_correct_angular_momentum_sign(self):
        x, h = make_grid(33, 8)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        psi = (X + 1j * Y) * np.exp(-0.5 * (X*X + Y*Y + Z*Z))
        info = angular_diagnostics(psi, x, h)
        self.assertLess(abs(info["Lz_mean"] - 1), 0.05)
        self.assertLess(info["Lz_std"], 0.05)


if __name__ == "__main__":
    unittest.main()
