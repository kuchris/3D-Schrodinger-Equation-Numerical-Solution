# Numerical hydrogen orbitals and readable plots

This guide describes the symmetry-aided mode, `method="symmetry"`. For solving
from the full Hamiltonian without the orbital rules, use `direct.cmd` and
[the full-grid guide](direct-hamiltonian.md).

Run `setup.cmd` to install the plotting dependency, then double-click
`hydrogen.cmd`. Edit `config/hydrogen.toml` to select a named orbital.
The default is `3d_z2`, using n=100 interior points and a cube of side 50.
It writes both central slices and a separate 3D isosurface PNG in `outputs/plots/`.

Supported labels: `1s`, `2s`, `3s`, `2p_x`, `2p_y`, `2p_z`, `3p_x`, `3p_y`,
`3p_z`, `3d_xy`, `3d_xz`, `3d_yz`, `3d_x2-y2`, and `3d_z2`.
These are real orbital orientations; real p/d labels need not be Lz eigenstates.
They are low-state symmetry selections, not a general quantum-number classifier.

## The numerical method is retained

The state is computed with the same three-dimensional Cartesian finite-difference
Hamiltonian, `H = -0.5 * Laplacian - 1/r`, and zero boundary on the cube faces.
No analytical hydrogen wavefunction is used to generate the plotted data.

`schrodinger3d/hydrogen.py` uses exact reflection symmetries to solve a smaller
matrix on the positive octant. On a centered even grid, the first point is h/2.
Its ghost neighbor at -h/2 equals `parity * psi(h/2)`, so each inner octant face
adds `-parity / (2*h**2)` to the existing kinetic diagonal. Mirroring the result
recovers a full-grid state of the original discrete Hamiltonian.

Reflection parities isolate real p and off-diagonal d sectors. Permuting x/y/z
and projecting onto the corresponding symmetry combinations selects s,
d(x2-y2), and d(z2) states. These operations commute with the discrete cubic
Hamiltonian. Tests also apply the original full-grid Hamiltonian to reconstructed
states and check the residual and normalization. Analytical functions appear
only in tests for comparison; the runtime reference uses only `E_n=-1/(2*n**2)`.

The cube and grid break continuous rotational symmetry. The labels describe
states approaching the corresponding free-atom orbitals on convergence;
they do not assert exact continuum L2 quantum numbers on a finite grid.

## Why the earlier excited-state plots differed

- `state` is an energy-sorted finite-box index, not a principal or angular
  quantum number. An arbitrary eigenvector in a degenerate space can have a
  different orientation or combination from a textbook drawing.
- L=8 is too small for free-atom excited orbitals. In a reproduced n=26 run,
  state 4 had E=+0.21144751. This is a valid finite-box eigenstate, but not a
  free-atom bound orbital. The generic plot now marks E>=0 explicitly.
- Central planes may be nodal planes. On an even grid, psi is interpolated
  to zero before taking its squared magnitude. Interpolating density instead
  would incorrectly fill a node. Almost-zero planes are labeled as nodes.
- Linear density colors can hide weaker outer lobes. Log density and auto zoom
  help visibility without changing the computed state.

## Display settings

```toml
[plot]
hydrogen_orbital = "3d_z2"
quantity = "wavefunction"
zoom = "auto"
density_scale = "log"
surface = true
```

`hydrogen_orbital = ""` restores the generic energy-index selection. A named
orbital uses its symmetry selector instead of `state` or the requested `states`.
It requires `potential = "hydrogen"` and rejects a nonnegative selected energy.

`zoom = "auto"` frames a centered cube containing at least 99.5% of the sampled
probability; `full` shows the whole computational box. Coordinates remain in the
same physical scale. `density_scale = "log"` shows density down to 1e-4 of the
peak. All three cuts still share a color scale.

The 3D plot uses signed-amplitude isosurfaces where density is 4% of the peak.
Red and blue mark signs, not electron trajectories or a hard orbital boundary.
Surface size depends on that display threshold.

## Accuracy checks

The default numerical 3d_z2 example gave E=-0.055615798 versus -1/18, a relative
difference of about 0.108%, with eigenpair residual about 1.6e-14. This is one
configuration, not a general accuracy guarantee. Grid and finite-box errors can
cancel in the energy. Refine the spacing and enlarge the box separately, inspect
the wavefunction, and monitor the reported outer-shell probability.

Sources: [hydrogen spectrum and separation of variables](https://farside.ph.utexas.edu/teaching/qmech/lectures/node82.html),
[marching cubes used for numerical isosurfaces](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes).
