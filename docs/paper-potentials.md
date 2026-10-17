# Potentials from the paper

Source: Wai Kui Wong, [Solving 3D Time Independent Schrodinger Equation Using
Numerical Method](https://doi.org/10.13140/RG.2.2.13878.01602/1), December 2021.
The formulas were checked against Section IV and Appendix A of the
[ResearchGate full text](https://www.researchgate.net/publication/357203546_Solving_3D_Time_Independent_Schrodinger_Equation_Using_Numerical_Method).

Choose a preset with `potential = "..."` in `config/settings.toml`. The code is
in [potentials.py](../schrodinger3d/potentials.py). In the table, `L` is the
cube side `length` and `rho = sqrt(x**2 + y**2)`.

| Setting | Formula | Where in the paper |
| --- | --- | --- |
| `box` | `0` inside the cube, with ψ = 0 on the faces | IV.A, p. 5 |
| `paper_harmonic` | `0.0001 * (x**2 + y**2 + z**2) / 2` | IV.B, Eq. (29), p. 10 |
| `circular` | `200 * (1 - square(2.1*pi*rho/L + (z/L)**2))` | IV.C; code in Appendix A, p. 58 |
| `torus` | `(rho/L - 0.4)**2 + (z/L)**2 - 0.1**2` | IV.D, Eq. (31), p. 21 |
| `tanh` | `tanh((x/L)**2 + (y/L)**2)` | IV.E, Eq. (32), p. 27 |
| `hydrogen` | `-1 / sqrt(x**2 + y**2 + z**2)` | IV.F, Eq. (33), p. 32 |

Two more options are not from the paper: `harmonic` is `(x**2 + y**2 + z**2) / 2`,
whose exact energies are known, and `custom` reads `custom/potential.py`.

## Notes on individual presets

- **Torus.** Section IV.D gives a ring radius R = 0.4L and a tube radius
  r = 0.1L, which are used here. The example in Appendix A uses R = 0.3L
  instead. The `-0.1**2` term only shifts every energy by the same amount; it
  does not change ψ or set the width of the ring.
- **Circular well.** Eq. (30) is garbled in the available text, so this preset
  copies the code printed in Appendix A. There, `square` is SciPy's periodic
  square wave, so V is either 0 or 400. The z term sits outside the square
  root, so this is not a simple spherical well. It has not been checked
  against the typeset Eq. (30).
- **Tanh.** Eq. (32) contains only x and y. Only the cube walls confine ψ in z.
- **Hydrogen.** The exact -1/r is used, with no smoothing. The grid must not
  contain r = 0, so an odd n is raised to the next even n, and the program says
  so. Calling `make_potential` directly on a grid that contains the origin
  raises an error.

## Why the energies differ from the paper

The formulas are the paper's, but the units are this program's (ħ = m = 1):

- The paper (Eq. (25) and Appendix A) scales its matrix by the grid spacing:
  `H_scaled = -0.5 * D3 + U`, with `U = h**2 * V`.
- This program uses `H = -0.5 * D3 / h**2 + V`.

So the same coefficient gives a different strength, and the paper's energies
and figures are not expected to be reproduced. An exact reproduction would also
need the paper's grid, boundaries, and scaling. Here the potential stays the
same function when you refine the grid.

Also keep in mind:

- `paper_harmonic` is very weak, so at a cube side of 8 the walls dominate.
- In the energy panel, only `box` and `harmonic` show exact reference energies.
- Check the grid spacing and the cube size before comparing any numbers.

For hydrogen, a reasonable first run is n = 40, a cube side of 20, and
state = 0, followed by a convergence check.
