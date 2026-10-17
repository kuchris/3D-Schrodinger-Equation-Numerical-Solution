# Full-grid mode (`direct.cmd`)

This mode solves the whole grid with no assumptions about orbital shapes.
Shapes are organised only after the solve. Run `direct.cmd` and edit
[config/direct.toml](../config/direct.toml). The default is hydrogen with
n = 64, a cube of side 40, and the 14 lowest states.

## What happens, step by step

1. **Solve.** Build the full matrix H, which has 64³ = 262,144 unknowns by
   default, and find the lowest states. No orbital names, n/l/m values, or
   symmetry rules go into this step.
2. **Sort equal-energy states**, if `basis = "symmetry"`. This is explained
   below.
3. **Measure**, if `angular_momentum = true`. Compute L², Lz, and mirror
   numbers for each state. These are read-outs and change nothing.
4. **Draw** the chosen `state` and print a table of all states.

## Why step 2 is needed

When several states have exactly the same energy, any mix of them also solves
Hψ = Eψ. H alone cannot pick one, so the eigensolver returns some arbitrary
mix, for example a p state that points diagonally instead of along x, y, or z.

Step 2 picks one definite set using symmetries of the potential itself:

- the mirrors x → -x, y → -y, and z → -z;
- the swap x ↔ y.

First, the program checks that V does not change under each operation, which
guarantees that the operation works together with H. Then, inside each group
of equal-energy states, it finds the combinations that each mirror either
keeps unchanged (+) or flips in sign (-). Finally, the x ↔ y swap separates
the one pair that the mirrors cannot. These signs are exactly how the textbook
real orbitals differ from one another.

Only states with the same energy are mixed, so every result is still an exact
solution of the same H. Energies do not change, and the residual stays around
10⁻¹⁴. No orbital table or formula is used. The names in the last column below
are read off afterwards from the measured signs and L².

Default run (n = 64, cube side 40):

| States | Energy | Mirror signs (x, y, z) | x ↔ y | Looks like |
| --- | --- | --- | --- | --- |
| 0 | -0.4629 | + + + | + | 1s |
| 1, 2, 3 | -0.1265 | - + +, + - +, + + - | ·, ·, + | 2p_x, 2p_y, 2p_z |
| 4 | -0.1216 | + + + | + | 2s |
| 5, 6, 7 | -0.0554 | - - +, - + -, + - - | +, ·, · | 3d_xy, 3d_xz, 3d_yz |
| 8, 9 | -0.0550 | + + + | -, + | 3d_x2-y2, 3d_z2 |
| 10, 11, 12 | -0.0548 | - + +, + - +, + + - | ·, ·, + | 3p_x, 3p_y, 3p_z |
| 13 | -0.0524 | + + + | + | 3s |

A dot (·) means the swap turns that state into another state of the same
group, for example p_x into p_y, so the state has no single swap sign.

With `basis = "raw"`, step 2 is skipped and the eigensolver's mixes are kept.

### Why mirrors, and not L² and Lz?

The textbook route is to use L² to separate s, p, and d, and then Lz to
separate m. On a cubic grid this does not work well:

- Inside one equal-energy group, every state has the same L², so L² separates
  nothing. On the grid, s, p, and d already have different energies.
- Lz cannot tell 3d_z2 from 3d_x2-y2 at all.
- States with m = ±2 would need a mix of two groups that have different
  energies on the grid (states 5-7 and 8-9). Such a mix is no longer a
  solution; in a test its residual rose to about 10⁻⁴.

The mirrors and the swap are exact symmetries of the grid, so they avoid all
three problems. L² and Lz are still measured, as checks.

### What can go wrong

- **A cut group.** If `states` ends in the middle of an equal-energy group,
  that group cannot be sorted. It is left unchanged and a NOTE is printed;
  increase `states`.
- **No symmetry.** If V has no mirror symmetry, for example a tilted custom
  potential, nothing is changed.
- **Axis choice.** Using the x ↔ y swap makes z the special axis, as textbooks
  do.

## Reading the angular-momentum numbers

With `angular_momentum = true`, the program prints these columns for every
state and saves them in `outputs/checks/<picture name>_analysis.json`:

| Column | Meaning |
| --- | --- |
| `<L2>` | Average of L². Values near 0, 2, and 6 suggest s, p, and d, because L² = l(l+1). |
| `L2 std` | Spread of L². A large spread means the state is not a clean l. |
| `l_eff` | The l that satisfies l(l+1) = `<L2>`. A diagnostic, not a label. |
| `<Lz>` | Always 0 here, because the computed states are real. It carries no information. |
| `Lz std` | Spread of Lz. About 0 for p_z and 3d_z2 (m = 0), about 1 for p_x and 3d_xz, about 2 for 3d_xy and 3d_x2-y2. |

The operators are built from finite differences, for example
Lz ψ = -i (x ∂ψ/∂y - y ∂ψ/∂x). On a grid they are approximate, especially near
the walls. For example, state 13 (3s) has `<L2>` = 0.25 with a spread of 3.4,
because the cube is small for it.

## Limits of the default run

- The grid is coarse: 1s comes out at -0.463 instead of -0.5.
- The cube is small for the n = 3 states: about 1-4% of their probability sits
  in the outer grid points next to the walls.
- A cubic grid splits energies that are equal in free space: 2s and 2p
  separate, and 3d splits into a group of 3 and a group of 2.

Sources: [Cartesian angular momentum](https://farside.ph.utexas.edu/teaching/sm1/Thermalhtml/node148.html),
[MIT notes on common eigenbases and degeneracy](https://www.ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/9ccc9fd132124dd102e1a7863e6e12ce_MIT8_04S13_OnCommEigenbas.pdf).
