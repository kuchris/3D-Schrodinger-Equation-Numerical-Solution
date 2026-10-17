"""Check numerical orbital selection, not analytical wavefunction rendering."""

import unittest
import numpy as np

from schrodinger3d.hydrogen import solve_hydrogen_orbital
from schrodinger3d.potentials import make_potential
from schrodinger3d.solver import build_hamiltonian
from schrodinger3d.display import central_wavefunction_slices, probability_view_limit


class HydrogenOrbitalChecks(unittest.TestCase):
    def test_p_orbital_uses_same_full_finite_difference_hamiltonian(self):
        x, h, energy, psi, info = solve_hydrogen_orbital(20, 20, "2p_z")
        H = build_hamiltonian(make_potential(x, "hydrogen", length=20), h)
        state = psi.ravel()
        residual = np.linalg.norm(H @ state - energy * state) / np.linalg.norm(state)
        self.assertLess(residual, 1e-8)
        np.testing.assert_allclose(psi, -psi[:, :, ::-1], atol=1e-12)
        np.testing.assert_allclose(psi, psi[::-1, :, :], atol=1e-12)
        self.assertAlmostEqual(np.sum(psi**2) * h**3, 1, places=10)
        self.assertLess(energy, 0)

    def test_d_z2_has_numerical_angular_shape_and_small_residual(self):
        x, h, energy, psi, info = solve_hydrogen_orbital(32, 40, "3d_z2")
        np.testing.assert_allclose(psi, psi.swapaxes(0, 1), atol=1e-11)
        mid = len(x) // 2
        distance = int(np.argmin(abs(x - 5)))
        self.assertLess(psi[mid, mid, distance] * psi[distance, mid, mid], 0)
        self.assertLess(info["relative_residual"], 1e-8)
        self.assertLess(abs(energy + 1 / 18), 0.01)

    def test_named_orbital_rejects_small_box_instead_of_claiming_bound_state(self):
        with self.assertRaisesRegex(ValueError, "not bound"):
            solve_hydrogen_orbital(20, 8, "3d_xy")

    def test_even_grid_interpolation_preserves_a_real_nodal_plane(self):
        x, h, energy, psi, info = solve_hydrogen_orbital(20, 30, "3d_xy")
        xy, xz, yz = central_wavefunction_slices(psi, x)
        self.assertGreater(abs(xy[0]).max(), 1e-5)
        np.testing.assert_allclose(xz[0], 0, atol=1e-12)
        np.testing.assert_allclose(yz[0], 0, atol=1e-12)

    def test_auto_zoom_keeps_probability_and_reduces_empty_space(self):
        x = np.linspace(-15, 15, 81)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        density = Z**2 * np.exp(-(X**2 + Y**2 + Z**2))
        limit = probability_view_limit(x, density)
        inside = np.abs(x) <= limit
        retained = density[np.ix_(inside, inside, inside)].sum() / density.sum()
        self.assertGreaterEqual(retained, 0.995)
        self.assertLess(limit, 5)

    def test_numerical_d_xy_overlaps_analytic_shape_used_only_as_test(self):
        x, h, energy, psi, info = solve_hydrogen_orbital(80, 50, "3d_xy")
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        reference = X * Y * np.exp(-np.sqrt(X**2 + Y**2 + Z**2) / 3)
        reference /= np.sqrt(np.sum(reference**2) * h**3)
        fidelity = abs(np.sum(reference * psi) * h**3)**2
        self.assertGreater(fidelity, 0.999)


if __name__ == "__main__":
    unittest.main()
