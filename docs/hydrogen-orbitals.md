# Named hydrogen orbitals (`hydrogen.cmd`)

In this mode you name the orbital you want, for example `3d_z2`, and the
program finds it directly. It uses symmetry to make the problem 8 times
smaller, so it can afford a finer grid than [full-grid mode](direct-hamiltonian.md).

Run `hydrogen.cmd` and set `hydrogen_orbital` in
[config/hydrogen.toml](../config/hydrogen.toml). The default is `3d_z2` with
n = 100 and a cube of side 50. The run saves the centre cuts and a 3D picture
in `outputs/plots/`.

Supported names: `1s`, `2s`, `3s`, `2p_x`, `2p_y`, `2p_z`, `3p_x`, `3p_y`,
`3p_z`, `3d_xy`, `3d_xz`, `3d_yz`, `3d_x2-y2`, and `3d_z2`.

## How it differs from full-grid mode

| | `hydrogen.cmd` | `direct.cmd` |
| --- | --- | --- |
| You give | An orbital name | Nothing about shapes |
| Symmetry is used | Before solving, to choose which states to search | After solving, to sort states that share an energy |
| Grid that is solved | One eighth of the cube | The whole cube |
| Result | One state | The lowest `states` states |

Both solve the same finite-difference H, -½∇² - 1/r with ψ = 0 on the walls.
Neither puts a textbook formula into ψ. On the same grid they give the same
3d_z2 state, which a test checks.

## How the shortcut works

1. **Mirror signs.** Each of these orbitals is either even (+) or odd (-) under
   each of the mirrors x → -x, y → -y, and z → -z. For example, 2p_z is
   (+, +, -). The program takes these signs from a small table in
   [hydrogen.py](../schrodinger3d/hydrogen.py).
2. **Solve one eighth of the cube.** Once the signs are known, ψ in the corner
   where x, y, and z are all positive determines ψ everywhere, so only that
   corner is solved. A grid point next to a mirror plane sees its mirror image
   as +ψ or -ψ, which adds -sign / (2h²) to the matrix diagonal there. This is
   exact, not an approximation.
3. **Separate shapes that share signs.** s, 3d_x2-y2, and 3d_z2 all have the
   signs (+, +, +). Swapping axes tells them apart: an s state is unchanged by
   every swap, and 3d_x2-y2 changes sign under x ↔ y. 3d_z2 is then built from
   the 3d_x2-y2 state by permuting its axes.
4. **Pick the right one.** Among the matching states, the table's rank selects,
   for example, 3s as the third s-like state.
5. **Mirror back** to the full cube, and check that the result solves the full H.

Steps 1, 3, and 4 use a table of known orbital symmetries, so this mode
assumes the shape you ask for. Use `direct.cmd` if you want nothing assumed.

If the selected state has E ≥ 0, the run stops with an error. Such a state is a
standing wave in the box, not a bound orbital; enlarge the cube.

## Settings

```toml
[plot]
hydrogen_orbital = "3d_z2"
quantity = "wavefunction"
zoom = "auto"
density_scale = "log"
surface = true
```

`hydrogen_orbital` replaces `state`. It requires `potential = "hydrogen"` and
`method = "symmetry"`, and it must be empty in the other modes. The display
settings are explained in the [README](../README.md#main-settings).

## Accuracy

The default `3d_z2` run gives E = -0.055616, compared with the exact
-1/18 = -0.055556, a difference of about 0.11%. Its residual is about 10⁻¹⁴.
This is one example, not a guarantee: refine the grid and enlarge the cube to
check. The printed outer-shell probability tells you how much of ψ is close to
the walls.

## Common surprises

- In `plot.cmd`, `state` is an energy order, not an orbital name. States that
  share an energy can come out as tilted mixes. Use this mode, or `direct.cmd`
  with `basis = "symmetry"`.
- In a small cube (for example a side of 8), excited states have E > 0. They
  are box states, not orbitals.
- A centre cut can lie exactly on a node. It is then labelled "Nodal plane".
- Faint outer lobes may need `density_scale = "log"` to be visible.

Sources: [hydrogen spectrum and separation of variables](https://farside.ph.utexas.edu/teaching/qmech/lectures/node82.html),
[marching cubes used for the 3D surfaces](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes).
