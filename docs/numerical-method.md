# How the solver works

This page follows one calculation from the potential to the picture. The same
steps appear in code in [examples/basic_usage.py](../examples/basic_usage.py).

## 1. The equation

The program looks for stationary states of one particle:

```text
H ψ = E ψ,    H = -½ ∇² + V(x, y, z)
```

The units are ħ = m = 1. The particle lives in a cube of side `length`, and
ψ = 0 on its six faces, like hard walls. There is no time evolution.

## 2. Put space on a grid

Each axis gets `n` equally spaced points inside the cube, with spacing
h = length / (n + 1). The walls themselves are not stored, because ψ is known
to be 0 there. So ψ becomes a list of n³ numbers, one per grid point.

## 3. Turn derivatives into differences

A second derivative at a point is estimated from its two neighbours:

```text
ψ''(x) ≈ [ψ(x - h) - 2 ψ(x) + ψ(x + h)] / h²
```

Doing this along x, y, and z builds the Laplacian from each point and its six
neighbours. A neighbour outside the cube counts as 0. The potential is simply
multiplied point by point, so V is used as given, in energy units.

The result is a large matrix H of size n³ × n³ that is almost entirely zeros
(a sparse matrix). With n = 64 there are 262,144 unknowns. In code this is
`build_hamiltonian` in [solver.py](../schrodinger3d/solver.py).

## 4. Find the lowest energies

`solve_lowest_states` asks SciPy's `eigsh` for the eigenvalues with the lowest
values, negative ones included. Internally it asks for 8 more than requested,
which helps it find every member of a group of equal energies, and then keeps
the requested number.

Each ψ is scaled so that the total probability is 1:

```text
sum of |ψ|² × h³ = 1
```

States with equal energies come back as an arbitrary mix. Full-grid mode can
sort them afterwards; see [full-grid mode](direct-hamiltonian.md). If the
requested number of states ends in the middle of such a group, the group is
cut, so ask for a few more states than you need.

## 5. Check the answer

For every state the program reports a residual, the size of Hψ - Eψ relative
to ψ. A value around 10⁻¹⁴ means the matrix equation is solved. It does not
mean that the grid is fine enough or the cube big enough.

To check the physics, repeat with a smaller spacing at the same `length`, and
with a bigger `length` at the same spacing. Trust the digits that stop
changing. Look at ψ as well, because errors in the energy can cancel.

## 6. Draw the result

[display.py](../schrodinger3d/display.py) and [plotting.py](../schrodinger3d/plotting.py)
change only how ψ is shown, never ψ itself.

- **Centre cuts.** With an even n there is no grid point at 0, so ψ is
  interpolated to 0 first and squared afterwards. Squaring first would wrongly
  fill in a node. A cut where ψ is almost 0 everywhere is labelled as a nodal
  plane.
- **Auto zoom** chooses the smallest centred cube that holds at least 99.5% of
  the probability.
- **Log density** shows values down to 10⁻⁴ of the peak.
- **3D surface** is drawn where the density equals 4% of its peak. Red and
  blue show the sign of ψ.

## Array layout (for programmers)

| Quantity | Shape |
| --- | --- |
| Coordinates | (n,) |
| Potential | (n, n, n) |
| H | (n³, n³), sparse |
| Energies | (k,) for k states |
| Wavefunctions | (n³, k), one state per column |

Arrays use C order, so z changes fastest. To get state i as a 3D array, use
`wavefunctions[:, i].reshape(n, n, n)`.

The hydrogen symmetry shortcut used by `hydrogen.cmd` is explained in
[named hydrogen orbitals](hydrogen-orbitals.md).
