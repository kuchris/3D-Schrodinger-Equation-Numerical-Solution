# From a potential function to a numerical orbital

## 1. Define the physical problem

The solver uses model units with hbar=mass=1:

```text
H psi = E psi
H = -0.5 * Laplacian + V(x,y,z)
```

`schrodinger3d/potentials.py` supplies the preset functions. The user function
is in `custom/potential.py`. Hydrogen uses -1/r without softening. The model
is a single nonrelativistic particle in a finite cube with zero wavefunction
on its six faces. These boundary conditions affect extended states.

## 2. Replace derivatives by finite differences

For n unknown interior points on each axis, h=L/(n+1). The known zero values
at the cube faces are outside the unknown vector. The second derivative is:

```text
psi_xx(i,j,k) = [psi(i-1,j,k) - 2*psi(i,j,k) + psi(i+1,j,k)] / h**2
```

The y and z terms have the same form. Kronecker sums assemble the seven-point
3D stencil into a sparse matrix. V is added as a diagonal matrix. The input
potential and output energy have the same energy units; V is not multiplied
by another grid-spacing factor.

## 3. Solve the discrete eigenproblem

`solve_lowest_states` calls `eigsh` with `which="SA"`, selecting the lowest
algebraic energies, including negative values. It searches beyond the requested
cutoff to help resolve repeated energies, then sorts and keeps the requested
states. This is not a proof of complete degeneracy recovery.

Each state is normalized with the volume element:

```text
sum(abs(psi)**2) * h**3 = 1
```

Arrays use C order, with z varying fastest:

| Quantity | Shape |
| --- | --- |
| Coordinates | (n,) |
| Potential | (n,n,n) |
| Sparse Hamiltonian | (n**3,n**3) |
| Energies | (number_of_states,) |
| Wavefunctions | (n**3,number_of_states), one state per column |

Generic `state=4` means the fifth returned eigenvector. It does not mean a
principal quantum number of four, or specify a p/d orbital orientation.

## 4. Optional hydrogen symmetry selection

This step is only used for the named hydrogen mode. It still solves the same
Cartesian finite-difference Hamiltonian. It does not insert an analytical R(r)
or spherical harmonic into the plotted wavefunction.

The Coulomb potential and centered cubic box are invariant under reflections
and coordinate permutations. Their operators commute with the discrete H.
We therefore select eigenstates in an appropriate symmetry sector.

For example, a p_z-like state is even in x and y and odd in z:

```text
psi(-x,y,z) = +psi(x,y,z)
psi(x,-y,z) = +psi(x,y,z)
psi(x,y,-z) = -psi(x,y,z)
```

On an even grid, solve on the positive octant and mirror the values with these
signs. This reduces the unknown count by eight. At the first point h/2 on an
axis, the ghost value is sigma*psi(h/2), where sigma is +1 or -1. The stencil is:

```text
psi_xx(h/2) = [psi(3h/2) + (-2+sigma)*psi(h/2)] / h**2
```

Relative to the ordinary kinetic diagonal, this adds -sigma/(2*h**2).
This is an exact restriction of the original discrete matrix to that reflection
sector. Reconstructing the full cube and dividing by sqrt(8) preserves the norm.

For s and selected real d orientations, coordinate permutations project the
computed eigenvectors into a suitable cubic symmetry representation. For
example, subtracting the x/y-swapped vector isolates an x2-y2-like combination;
permuting and combining that numerical vector gives the z2-like orientation.
The low supported labels use the ordering within these sectors to select the
radial excitation. This is not a general classifier for all n,l,m states.

A finite cube has cubic rather than full rotational symmetry. The selected
states approach the named free-atom orbitals as grid and box errors are reduced;
they are not exact continuum L2 eigenstates on every finite grid.

## 5. Draw the computed data

`schrodinger3d/plotting.py` orchestrates the calculation; `display.py` handles
the display operations:

- Central cuts interpolate psi to zero before taking abs(psi)**2. This preserves
  an odd orbital's nodal plane. An almost-zero plane is labeled as a node.
- Auto framing retains at least 99.5% of the probability sampled inside the
  computational box. It changes axis limits, not the wavefunction.
- Log density reveals values down to 1e-4 of the sampled peak. It changes the
  color mapping, not the density.
- A 3D isosurface marks density equal to 4% of its peak. Red/blue encode the
  signs of psi. This surface is a display threshold, not a hard orbital edge.

## 6. Check numerical and physical accuracy separately

The tests reconstruct symmetry-selected states and apply the original full-grid
Hamiltonian. They check eigenpair residuals, parity, normalization, nodal planes,
and the displayed probability fraction. An analytical orbital appears only in
a comparison test, never in the runtime wavefunction generation.

Small residuals mean the matrix eigenproblem is solved accurately. Physical
accuracy also requires fixing L while reducing h, then fixing h while enlarging
L. Energy errors can cancel, so inspect the wavefunction as well. An E>=0
Coulomb box state is not a free-atom bound orbital. Named mode rejects it.

See [hydrogen details and measured example](hydrogen-orbitals.md) and
[paper formula sources](paper-potentials.md).
