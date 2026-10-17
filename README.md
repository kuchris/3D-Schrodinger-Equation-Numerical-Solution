# 3D stationary Schrödinger solver

This program finds the stationary states of one particle in a 3D potential.
It solves

```text
H ψ = E ψ,    H = -½ ∇² + V(x, y, z)
```

inside a cube, in units where ħ = m = 1. You choose the potential V. The
program computes the lowest energies E and their wavefunctions ψ on a grid,
then draws them.

![Numerical finite-difference hydrogen 3d_z2 orbital](docs/images/hydrogen-3d-z2.png)

## Quick start (Windows)

1. Install Python 3.10 or newer, then double-click `setup.cmd` once.
2. Double-click one of the launchers below.
3. Open the pictures in `outputs/plots/`.

| Launcher | Settings file | What it does |
| --- | --- | --- |
| `plot.cmd` | `config/settings.toml` | Any potential. Draws one state, chosen by energy order. |
| `direct.cmd` | `config/direct.toml` | Hydrogen on the whole grid, with no orbital shapes assumed. States that share an energy are then sorted into clean shapes. |
| `hydrogen.cmd` | `config/hydrogen.toml` | One named hydrogen orbital, such as `3d_z2`, found with a symmetry shortcut. |

To change a run, edit its settings file, save, and double-click the launcher
again. Comments in each settings file explain the options. A misspelled
setting stops the run with a message that names it.

## Which launcher should I use?

- **My own potential, or one from the paper:** `plot.cmd`.
- **Hydrogen, with nothing assumed about orbital shapes:** `direct.cmd`.
  It solves the full grid first. Only afterwards does it organise states that
  share an energy.
- **One specific hydrogen orbital on a finer grid:** `hydrogen.cmd`.
  You name the orbital, and the program searches only states with that
  orbital's symmetry. This makes the problem 8 times smaller.

On the same grid, `direct.cmd` and `hydrogen.cmd` give the same 3d_z2 state
(checked by a test to 8 decimal places).

## Three ideas that make the results easier to read

1. **`state` counts energy levels, not quantum numbers.** `state = 0` is the
   lowest energy and `state = 4` is the fifth. States with equal energy are
   counted separately. It is not n, l, or m.
2. **Equal energies do not fix a shape.** When several states share one energy,
   such as the three 2p states, any mix of them is also a solution. The solver
   may return a tilted mix. With `basis = "symmetry"`, `direct.cmd` re-mixes
   them, only within that energy, into the shapes that the potential's mirror
   symmetries pick out. That is where p_x, p_y, and p_z come from.
   See [full-grid mode](docs/direct-hamiltonian.md).
3. **Display options do not change the data.** Zoom, log colours, and 3D
   surfaces change only how the same ψ is drawn.

## Your own potential

Edit `V` in `custom/potential.py`:

```python
def V(x, y, z):
    return 0.2 * (x**2 - 4)**2 + 0.5 * (y**2 + z**2)
```

Then set `potential = "custom"` in `config/settings.toml` and run `plot.cmd`.

- x, y, and z are NumPy arrays, so use `np.exp`, `np.where`, and so on.
- Return real, finite numbers. A single number also works.
- Give V directly in energy units. Do not multiply it by the grid spacing.
- ψ is forced to zero on the six faces of the cube, so make the cube big enough.

## Built-in potentials

| `potential =` | Meaning |
| --- | --- |
| `box` | Empty cube (particle in a box) |
| `harmonic` | r²/2, with exact energies 1.5, 2.5, ... for comparison |
| `paper_harmonic` | The paper's weak harmonic well, Eq. (29) |
| `circular` | The paper's circular well, Appendix A |
| `torus` | The paper's ring-shaped well, Eq. (31) |
| `tanh` | The paper's tanh potential, Eq. (32), which does not depend on z |
| `hydrogen` | Coulomb potential -1/r, Eq. (33) |
| `custom` | Your function in `custom/potential.py` |

The paper presets use the paper's formulas in this program's units, so their
energies are not meant to match the paper's numbers. Formulas, page
references, and the unit difference are in [paper potentials](docs/paper-potentials.md).

## Main settings

| Setting | Meaning |
| --- | --- |
| `n` | Grid points per axis. The cost grows like n³. Hydrogen needs an even n, so an odd n is raised by one. |
| `length` | Side of the cube. Each axis runs from -length/2 to +length/2. |
| `states` | How many of the lowest states to compute. At least `state + 1` are computed. |
| `state` | Which state to draw. 0 is the lowest energy. |
| `quantity` | `"wavefunction"` shows the sign of ψ in red and blue; `"density"` shows \|ψ\|². |
| `zoom` | `"auto"` frames 99.5% of the probability; `"full"` shows the whole cube. |
| `density_scale` | `"log"` makes faint lobes visible. Density pictures only. |
| `surface` | `true` also saves a 3D picture. |
| `output` | Empty means an automatic file name in `outputs/plots/`. |

`direct.toml` adds `angular_momentum`, which prints L² and Lz measurements and
saves a report in `outputs/checks/`, and `basis`, described above.
`hydrogen.toml` adds `hydrogen_orbital`.

## Reading the pictures

- Three panels cut through the centre of the cube, at z = 0, y = 0, and x = 0.
- Red is positive ψ, blue is negative, and white is zero. The overall sign is
  arbitrary.
- The fourth panel lists the computed energies. The star marks the drawn state.
- The 3D surface is drawn where the density is 4% of its peak. It is a display
  threshold, not the edge of the orbital.

## How accurate is it?

For every state the program checks that ψ really solves the grid equation. The
reported residual is around 10⁻¹⁴. That is not the same as matching the
continuous physics, because two errors remain:

- **The grid is too coarse.** For example, hydrogen 1s on the `direct.toml`
  grid (spacing 0.62) gives -0.463 instead of -0.5.
- **The cube is too small.** Large orbitals get squeezed by the walls.

To trust a number, shrink the spacing at a fixed `length`, then enlarge
`length` at a fixed spacing, and keep the digits that stop changing. A cubic
grid can also split energies that are equal in free space slightly; for
example, 3d becomes a group of 3 states and a group of 2.

## Command line (optional)

Run these from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m schrodinger3d.plotting
.\.venv\Scripts\python.exe -B -m schrodinger3d.plotting --config config/direct.toml --state 3
.\.venv\Scripts\python.exe -B -m examples.basic_usage
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

A command-line option overrides the settings file for that run; `--help` lists
them. On macOS or Linux, create the environment with `python3 -m venv .venv`
and use `.venv/bin/python`.

## More detail

| Document | Read it for |
| --- | --- |
| [How the solver works](docs/numerical-method.md) | Grid, matrix, eigenvalues, and pictures, step by step |
| [Full-grid mode](docs/direct-hamiltonian.md) | `direct.cmd`, sorting equal-energy states, angular-momentum numbers |
| [Named hydrogen orbitals](docs/hydrogen-orbitals.md) | `hydrogen.cmd` and its symmetry shortcut |
| [Paper potentials](docs/paper-potentials.md) | Formulas, sources, and unit differences |
| [Folder layout](docs/folder-layout.md) | Where each file lives and what each module does |

A short complete calculation in code is in [examples/basic_usage.py](examples/basic_usage.py).
The original scripts are on the `legacy` branch.

Related paper: [Solving 3D Time Independent Schrodinger Equation Using
Numerical Method](https://doi.org/10.13140/RG.2.2.13878.01602/1).
