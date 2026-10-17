# Potential presets from the paper

Source: Wai Kui Wong, [Solving 3D Time Independent Schrodinger Equation Using
Numerical Method](https://doi.org/10.13140/RG.2.2.13878.01602/1), December 2021.
The accessible [ResearchGate full text](https://www.researchgate.net/publication/357203546_Solving_3D_Time_Independent_Schrodinger_Equation_Using_Numerical_Method)
was checked against Section IV and Appendix A.

Select a preset by changing `simulation.potential` in `settings.toml`.
Here `L` is `simulation.length`, and `rho = sqrt(x**2 + y**2)`.

| Setting | Formula | Paper location |
| --- | --- | --- |
| `box` | `0` inside the cube, with zero wavefunction on the faces | IV.A, p. 5 |
| `paper_harmonic` | `0.0001 * (x**2 + y**2 + z**2) / 2` | IV.B, Eq. (29), p. 10 |
| `circular` | `200 * (1 - square(2.1*pi*rho/L + (z/L)**2))` | IV.C; executable SP expression in Appendix A, p. 58 |
| `torus` | `(rho/L - 0.4)**2 + (z/L)**2 - 0.1**2` | IV.D, Eq. (31), p. 21 |
| `tanh` | `tanh((x/L)**2 + (y/L)**2)` | IV.E, Eq. (32), p. 27 |
| `hydrogen` | `-1 / sqrt(x**2 + y**2 + z**2)` | IV.F, Eq. (33), p. 32 |

The existing `harmonic` option remains the separate `omega=1` benchmark,
`V = (x**2 + y**2 + z**2) / 2`. `custom` still reads `custom_potential.py`.

## Source details

- Torus: Section IV.D specifies `R=0.4L` and `r=0.1L`; these values are used.
  Appendix A's active example instead uses `R=0.3L`. The latter is not the
  default for the paper preset.
- Circular well: Eq. (30) is garbled in the accessible text extraction.
  This preset therefore follows the explicit `scipy.signal.square` expression
  printed in Appendix A. The `z**2` term is outside the square root. It is not
  silently replaced with a conventional spherical step potential, and equivalence
  to the typeset Eq. (30) has not been visually confirmed. `square` means the
  periodic square-wave function; the returned potential takes values 0 and 400.
- Tanh: Eq. (32) contains x and y only. The preset retains this z independence;
  the cube boundary still confines the wavefunction in z.
- Torus: the `-0.1**2` term shifts the energy zero. That constant does not
  change the eigenfunctions or set their transverse width.

## Units and reproduction limits

These are the paper's potential functions evaluated directly in the new
solver's model energy units, with `hbar=mass=1`. They are not a claim to
reproduce the paper's original numerical spectra or figures.

The paper's Eq. (25) and Appendix A instead use a grid-scaled Hamiltonian:
`H_scaled = -0.5 * D3 + U`, with `U = h**2 * V` for mass=1. This solver uses
`H = -0.5 * D3 / h**2 + V`. Copying a numerical U coefficient directly into V
does not preserve the old kinetic-to-potential ratio. A strict reproduction
would also need the original grid, boundaries, scaling, and energy selection.
The presets retain fixed functions when refining the grid in this solver.

The weak `paper_harmonic` case is strongly affected by the finite box at L=8.
For all cases, check both grid spacing and box size before physical comparisons.
Only `box` and the omega=1 `harmonic` preset currently show analytical reference
curves; the other plots show computed levels without an assumed exact spectrum.

## Coulomb singularity

The hydrogen preset keeps the exact point-sampled `-1/r` formula. It adds no
softening parameter or arbitrary finite value at the origin. The app changes
an odd n to the next even n and reports the adjustment. This gives a symmetric
interior grid that does not contain the origin. Calling `make_potential`
directly on a grid containing the origin raises a clear error.

For example, start with n=40, L=20, and state=0, then check convergence. These
numbers are an example configuration, not a certified accuracy threshold.
