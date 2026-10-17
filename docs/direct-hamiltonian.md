# Full-Hamiltonian mode: solve first, measure afterwards

Double-click `direct.cmd`. Edit `config/direct.toml` to select the potential,
grid, count of eigenstates, and energy index to display.

The generic solver already supported full-grid solving. This dedicated mode
makes the choice explicit and adds post-solve angular diagnostics and a larger
hydrogen example. It is independent of the named-orbital symmetry mode.

## Inputs and calculation

```toml
[simulation]
method = "full"
potential = "hydrogen"
n = 64
length = 40.0
states = 14

[plot]
state = 9
hydrogen_orbital = ""
quantity = "wavefunction"
zoom = "auto"
surface = true

[analysis]
angular_momentum = true
```

`schrodinger3d/direct.py` constructs the full Cartesian grid and the original
sparse `H = -0.5 * Laplacian + V`. It passes only H, the state count, and volume
spacing to `solve_lowest_states`. The eigensolver selects the lowest algebraic
energies from a seeded random starting vector.

There is no n/l/m input, spherical harmonic, orbital template, parity sector,
octant solve, or rotation of an eigenspace in this route. For n=64 the matrix
has 262144 unknowns. `state=9` means the tenth energy-sorted vector; it does
not request a particular orbital shape.

After the eigensolver returns, the measurement pass reads each eigenvector
without modifying it. Tests verify full-grid dimensions, agreement with an
independent dense eigensolve for an asymmetric potential, and immutability
under measurement. The full route does not import the named-orbital module.

## Angular measurements

In units hbar=1 the Cartesian operators are:

```text
Lx psi = -i * (y * d(psi)/dz - z * d(psi)/dy)
Ly psi = -i * (z * d(psi)/dx - x * d(psi)/dz)
Lz psi = -i * (x * d(psi)/dy - y * d(psi)/dx)
L2 psi = Lx(Lx psi) + Ly(Ly psi) + Lz(Lz psi)
```

`angular.py` uses centered first differences and coordinate multiplication.
It reports <L2>, its standard deviation, <Lz>, its standard deviation, and
measured reflection expectations. These are measurements of the output,
not constraints used to obtain it.

The continuous diagnostic `l_effective = (sqrt(1+4*<L2>)-1)/2` is included for
inspection. It is not a guaranteed quantum-number classification. In particular,
<Lz>=0 with a nonzero spread does not establish a definite m=0 state.

## What is and is not determined

H alone determines energies and energy eigenspaces. A degenerate eigenspace
does not specify a unique p_x, p_y, p_z, or n/l/m basis. The raw returned vectors
can differ in orientation or combination between implementations while having
the same energy. This mode preserves those original eigenvectors.

The finite cube and Cartesian discretization do not preserve exact continuum
rotational symmetry. Angular diagnostics, especially L2 spread near a hard
boundary, can therefore have substantial discretization and boundary effects.
L2 spread is computed by applying the discrete angular operators twice with
zero exterior values; it must not be treated as an exact continuum measurement.

For example, the n=64,L=40 run produced a state with <L2>=5.9822 and l_effective
about 1.9964, but L2 standard deviation about 1.3644. A mean close to 6 is
insufficient to certify a pure l=2 state. Enlarge the domain and refine the grid
separately before drawing physical conclusions. The default is an executable
demonstration, not an accuracy threshold.

This implementation does not silently assign n/l/m or rotate nearly degenerate
vectors into textbook shapes. Such a rotation would require additional choices
and checking its residual against the finite-grid Hamiltonian.

## Outputs and comparison

- `outputs/plots/`: cuts and optional 3D isosurface of the selected raw state.
- `outputs/checks/*_analysis.json`: all energies, residuals, probability checks,
  angular moments/spreads, and flags confirming full-grid unrotated solving.
- `plot.cmd`: general full-grid mode, usually without angular reports.
- `hydrogen.cmd`: the older named-orbital mode with `method="symmetry"`.

Sources: [Cartesian angular momentum](https://farside.ph.utexas.edu/teaching/sm1/Thermalhtml/node148.html),
[MIT notes on common eigenbases and degeneracy](https://www.ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/9ccc9fd132124dd102e1a7863e6e12ce_MIT8_04S13_OnCommEigenbas.pdf).
