# 3D stationary Schrödinger solver

A Python solver for the stationary 3D Schrodinger equation using second-order
finite differences, sparse Kronecker sums, and `eigsh`. Define your own real
potential, select an energy state in a settings file, and plot its wavefunction
or probability density.

Related paper: [Solving 3D Time Independent Schrodinger Equation Using Numerical
Method](https://doi.org/10.13140/RG.2.2.13878.01602/1).

## Windows quick start

Requires Python 3.10 or later on PATH.

1. Double-click `setup.cmd` once to create `.venv` and install dependencies.
2. Edit `custom_potential.py` to define `V(x, y, z)`.
3. Edit `settings.toml` to select the grid, state, and plot type.
4. Double-click `plot.cmd`. Open the generated image in `plots/`.

![Example custom-potential eigenstate](docs/images/custom-state.png)

The screenshot uses the initial custom harmonic function, state 4, and a 25^3
grid. Custom functions do not get an assumed analytical reference curve.

## Run the solver

Alternatively, install and run from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe schrodinger_3d.py --potential harmonic
.\.venv\Scripts\python.exe schrodinger_3d.py --potential box --n 24 --length 8 --states 6
.\.venv\Scripts\python.exe -m unittest -v test_physics
```

On macOS or Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python plot_states.py
```

## Plot using the settings file

1. Edit and save `settings.toml`.
2. Double-click `plot.cmd`.
3. Open the generated image in `plots/`.

The current settings select state 4 of your custom potential and display its signed
wavefunction. The launcher uses the local `.venv` created by `setup.cmd`.
It keeps the console open so you can read the result or any error.

```toml
[simulation]
potential = "custom"
n = 25
length = 8.0
states = 10

[plot]
state = 4
quantity = "wavefunction"
output = ""
```

Every setting has an English comment in `settings.toml`. An empty `output`
selects an automatic filename. Relative output paths are resolved from the
settings file's directory, regardless of the working directory at launch.

You can also run the plotter without command-line arguments:

```powershell
.\.venv\Scripts\python.exe plot_states.py
```

Command-line options override individual settings for one run:

```powershell
.\.venv\Scripts\python.exe plot_states.py --potential harmonic --state 0 --quantity density
.\.venv\Scripts\python.exe plot_states.py --potential box --state 0 --quantity density
```

Use `--config another_settings.toml` to select a different settings file.
The settings file only controls `plot_states.py`; the standalone solver keeps
its own command-line options.

Each image shows the selected state's central XY, XZ, and YZ slices, plus
computed energies. Built-in potentials also show continuum reference levels;
custom potentials have no assumed analytical reference. All three slices share
one color scale. These are local slices, not integrated projections.
The default odd grid size, `n=25`, includes zero exactly; an even grid uses the
nearest available coordinate and labels its actual position.

Plotting lives in `plot_states.py`, keeping it separate from the solver.

## Excited states

`state` is an index in the energy-sorted list: 0 is the ground state and 1 is
the first excited state. It is not a quantum number for an individual axis.
`states` sets the minimum number of eigenstates to compute. Selecting a higher
index automatically increases that count to at least `state + 1`.

```powershell
.\.venv\Scripts\python.exe plot_states.py --potential harmonic --state 1 --quantity wavefunction
.\.venv\Scripts\python.exe plot_states.py --potential harmonic --state 4 --quantity wavefunction
.\.venv\Scripts\python.exe plot_states.py --potential harmonic --state 4 --quantity density
```

`wavefunction` uses red and blue for positive and negative amplitude, with white
at zero. The overall sign of a real eigenfunction is arbitrary.
`density` displays `abs(psi)**2`, which is nonnegative.

For degenerate energies, the solver can return linear combinations within the
degenerate subspace rather than states with definite `(nx, ny, nz)` labels.
A finite grid can also split degeneracies of the continuum model.
A central slice may lie on a nodal plane; a zero slice does not mean the entire
state is zero. Higher states generally need finer grids and, for the harmonic
model, a sufficiently large box.

## Read the code

Follow `main()` from top to bottom:

1. `make_grid`: construct the interior grid points.
2. `make_potential`: define the potential at each point.
3. `build_hamiltonian`: assemble the kinetic and potential energy matrices.
4. `solve_lowest_states`: compute, sort, and normalize the lowest eigenstates.
5. Check energies, total probability, and eigenpair residuals.

## Units and conventions

- Model units set `hbar = mass = 1`; these are not SI units.
- `H = -0.5 * Laplacian + V`, so the input potential and output energies share
  the same energy scale.
- `1 / spacing**2` is included in the Laplacian. Do not apply another grid
  scaling factor to the potential.
- `n` counts interior points per axis; spacing is `length / (n + 1)`.
- The six cube faces are at `+/- length/2`, with `psi = 0` on each face.
- `indexing="ij"` matches C-order flattening and reshaping; the third axis
  varies fastest.
- `which="SA"` selects the lowest algebraic energies, including negative ones.
- Each wavefunction satisfies `sum(abs(psi)**2) * spacing**3 = 1`.

`harmonic` uses `V = (x**2 + y**2 + z**2) / 2`, with `omega = 1`.
Its infinite-space ground energy is 1.5; this calculation also has finite-box
and finite-spacing errors. For `box`, the boundary condition implements the
infinite walls and the interior potential is zero.

## Define your own potential

1. Open `custom_potential.py` and edit the return expression inside `V(x, y, z)`.
2. Select `potential = "custom"` in `settings.toml` (already selected).
3. Save the files and double-click `plot.cmd`.

For example, a double well along x with harmonic confinement along y and z:

```python
def V(x, y, z):
    return 0.2 * (x**2 - 2.0**2)**2 + 0.5 * (y**2 + z**2)
```

The initial editable example is the harmonic function, so you can compare it
with the built-in `harmonic` option before making changes. The file also includes
commented examples of a Gaussian well and a finite spherical well.

`x`, `y`, and `z` are NumPy coordinate arrays. Use `np.exp`, `np.sin`, `np.sqrt`,
and `np.where` for array operations. For piecewise potentials, use `np.where`
instead of a Python `if x > ...` statement.

Return a real numeric scalar or an array broadcastable to `(n, n, n)`.
Complex, nonnumeric, NaN, and infinite potentials are rejected. A singular
potential such as Coulomb at the origin needs separate treatment.
Use model energy units with `hbar = mass = 1`, without any grid-spacing factor.
The six cube faces still impose `psi = 0`, including for custom potentials.
The custom plot does not assume a harmonic or box analytical spectrum.

The standalone solver also accepts `--potential custom`.

## Inspect a wavefunction

`wavefunctions[:, state]` stores one eigenstate as a vector. Reshape it into space:

```python
psi = wavefunctions[:, 0].reshape(potential.shape, order="C")
probability_density = np.abs(psi)**2
```

Degenerate eigenfunctions may appear as different orthonormal combinations
between runs. A finite ARPACK calculation does not prove that every degeneracy
has been recovered. Increasing `n` increases the number of unknowns as `n**3`.
Keep the physical potential fixed and check grid convergence and box-size
convergence separately. A small residual verifies the discrete eigenproblem;
it does not establish convergence to the continuum solution.

## Validation

`test_physics.py` checks the pointwise stencil, exact discrete box spectrum,
volume normalization, negative energy shifts, and second-order convergence
to the continuum box energy. The solver handles the stationary equation
(TISE); time evolution is not implemented.
