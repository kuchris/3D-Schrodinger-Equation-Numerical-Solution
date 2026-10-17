"""Plot central slices of a selected eigenstate and the lowest energy levels.

Run without arguments to read settings.toml next to this script.
Command-line options override the settings file.
"""

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 compatibility.

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from potentials import POTENTIAL_LABELS, make_potential
from hydrogen_orbitals import ORBITALS, solve_hydrogen_orbital
from orbital_display import central_wavefunction_slices, probability_view_limit, plot_isosurfaces
from schrodinger_3d import (
    build_hamiltonian,
    make_grid,
    solve_lowest_states,
)


def reference_energies(kind, length, count):
    """Continuum reference levels, with degeneracies counted separately.

    Harmonic reference: infinite space, omega = 1.
    Box reference: Dirichlet cube of side length `length`.
    """
    if kind not in ("box", "harmonic"):
        return None  # No assumed exact spectrum for the other presets.
    # An axis quantum number above count cannot enter the first count states.
    modes = np.arange(count + 1)
    a, b, c = np.meshgrid(modes, modes, modes, indexing="ij")
    if kind == "harmonic":
        energies = a + b + c + 1.5
    else:
        energies = np.pi**2 / (2 * length**2) * (
            (a + 1)**2 + (b + 1)**2 + (c + 1)**2
        )
    return np.sort(energies.ravel())[:count]


def draw_energy_levels(ax, energies, kind, length, state, orbital_info=None):
    """Draw computed energies, an optional reference, and the selected state."""
    index = np.arange(len(energies))
    reference = reference_energies(kind, length, len(energies))
    if orbital_info:
        reference = np.array([orbital_info["reference_energy"]])
    if reference is not None:
        ax.plot(index, reference, "o", ms=9, mfc="none", mec="#197568", mew=1.5,
                label="Continuum reference")
    ax.plot(index, energies, "_", ms=14, mew=2.3, color="#c65c29",
            label="Finite difference")
    if state > 0:
        ax.plot(state, energies[state], "*", ms=13, color="#2454a0",
                label=f"Selected state {state}")
    ax.set(xlabel="State index (including degeneracy)", ylabel="Energy",
           title="Lowest energy levels", xticks=index)
    if orbital_info:
        ax.set(xlabel="Numerical orbital", title="Orbital energy",
               xticklabels=[orbital_info["label"]])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    annotation = f"State {state}: {energies[state]:.6f}"
    if orbital_info:
        annotation = f"{orbital_info['label']}: {energies[state]:.6f}"
    if reference is not None:
        error = abs(energies[state] - reference[state]) / abs(reference[state])
        annotation += f"\nReference: {reference[state]:.6f}\nRelative difference: {error:.3%}"
    else:
        annotation += "\nAnalytical reference: not supplied"
    if kind == "hydrogen" and energies[state] >= 0:
        annotation += "\nE >= 0: not a free-atom bound orbital"
    # Put the annotation inside the largest energy gap, away from the levels.
    all_levels = np.sort(np.r_[energies, reference]) if reference is not None else energies
    if len(energies) > 1:
        gap = int(np.argmax(np.diff(all_levels)))
        annotation_y = all_levels[gap] + 0.85 * np.diff(all_levels)[gap]
    else:
        padding = 0.15 * max(1, abs(energies[0]))
        ax.set_ylim(all_levels.min() - padding, all_levels.max() + padding)
        ax.set_xlim(-0.6, 0.6)
        annotation_y = all_levels.max() + 0.7 * padding
    ax.text(0.04, annotation_y,
            annotation,
            transform=ax.get_yaxis_transform(), va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9})


def plot_summary(coordinates, spacing, length, energies, wavefunctions, kind, output,
                 state=0, quantity="density", zoom="full", density_scale="linear",
                 orbital_info=None):
    """Use one color scale for all three cuts of the selected state."""
    n = len(coordinates)
    psi = wavefunctions[:, state].reshape((n, n, n), order="C").copy()
    # The real eigenvector's overall sign is arbitrary; fix it for display.
    if psi.flat[np.argmax(np.abs(psi))] < 0:
        psi = -psi
    values = np.abs(psi)**2 if quantity == "density" else psi
    # Add the known zero values at the box faces before drawing contours.
    plot_coordinates = np.r_[-length / 2, coordinates, length / 2]
    slices = central_wavefunction_slices(psi, coordinates)
    view_limit = probability_view_limit(coordinates, abs(psi)**2) if zoom == "auto" else length / 2

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12, 10), layout="constrained", facecolor="#faf9f6")
    grid = fig.add_gridspec(2, 2)
    density_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                    fig.add_subplot(grid[1, 0])]
    if quantity == "density":
        if density_scale == "log":
            levels = np.geomspace(values.max() * 1e-4, values.max(), 40)
            norm = LogNorm(vmin=levels[0], vmax=levels[-1])
        else:
            levels = np.linspace(0, values.max(), 35)
            norm = None
        cmap = "magma"
        color_label = rf"Density $|\psi_{{{state}}}|^2$"
    else:
        norm = None
        limit = np.max(np.abs(values))
        levels = np.linspace(-limit, limit, 41)
        cmap = "RdBu_r"
        color_label = rf"Wavefunction $\psi_{{{state}}}$ (real)"

    for ax, (amplitude, xlabel, ylabel) in zip(density_axes, slices):
        plane = abs(amplitude)**2 if quantity == "density" else amplitude
        plot_values = np.pad(plane, 1).T
        if norm:
            plot_values = np.ma.masked_less_equal(plot_values, 0)
            ax.set_facecolor(plt.get_cmap(cmap)(0))
        # Transpose: array's first coordinate belongs on the horizontal axis.
        contour = ax.contourf(
            plot_coordinates, plot_coordinates, plot_values,
            levels=levels, cmap=cmap, norm=norm, extend="min" if norm else "neither",
        )
        if quantity == "wavefunction" and plane.min() < 0 < plane.max():
            ax.contour(plot_coordinates, plot_coordinates, np.pad(plane, 1).T,
                       levels=[0], colors="#555555", linewidths=0.6)
        fixed_axis = next(axis for axis in "xyz" if axis not in (xlabel, ylabel))
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=f"{xlabel.upper()}{ylabel.upper()} plane | {fixed_axis} = 0",
               xlim=(-view_limit, view_limit), ylim=(-view_limit, view_limit))
        if np.max(abs(amplitude)) < 1e-8 * np.max(abs(psi)):
            ax.text(0.5, 0.5, "Nodal plane\nAmplitude approximately zero",
                    ha="center", va="center", transform=ax.transAxes,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
        ax.set_aspect("equal")

    if quantity == "density" and density_scale == "log":
        color_label += " (log scale)"
    if orbital_info:
        color_label = "Probability density" if quantity == "density" else "Signed amplitude"
        if quantity == "density" and density_scale == "log":
            color_label += " (log scale)"
    colorbar_options = {}
    if norm:
        colorbar_options = {"ticks": values.max() * 10.0**np.arange(-4, 1), "format": "%.1e"}
    fig.colorbar(contour, ax=density_axes, shrink=0.75, label=color_label, **colorbar_options)

    energy_axis = fig.add_subplot(grid[1, 1])
    draw_energy_levels(energy_axis, energies, kind, length, state, orbital_info)

    name = POTENTIAL_LABELS[kind]
    title = "Probability density" if quantity == "density" else "Signed wavefunction"
    selection = orbital_info["label"] if orbital_info else f"State {state}"
    fig.suptitle(f"{name} | {selection} | {title}", fontsize=17)
    interpretation = (
        "Slices show local density, not integrated probability. "
        if quantity == "density" else
        "Red / blue: positive / negative amplitude; white: zero. Overall sign is arbitrary. "
    )
    reference_note = (
        "Numerical symmetry sector; reference is the infinite-space hydrogen energy."
        if orbital_info else
        {"harmonic": "Harmonic reference assumes infinite space.",
         "box": "Box reference uses the same cube boundaries.",
         "custom": "Custom potential; no analytical reference supplied."}.get(
            kind, "Paper formula in model units; no analytical reference supplied."
         )
    )
    fig.supxlabel(
        f"hbar = mass = 1 | interior grid {n} x {n} x {n} | "
        f"L = {length:g} | h = {spacing:.4f} | zero boundary at the cube faces\n"
        + f"View: +/-{view_limit:.3g}; central psi interpolated to zero | " + interpretation
        + "\n" + reference_note,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def read_settings(path):
    """Read the two small TOML tables and reject misspelled settings."""
    with path.open("rb") as file:
        settings = tomllib.load(file)

    fields = {
        "simulation": {"potential": str, "n": int, "length": (int, float), "states": int},
        "plot": {"state": int, "quantity": str, "output": str, "zoom": str,
                 "density_scale": str, "surface": bool, "hydrogen_orbital": str},
    }
    defaults = {}
    for section, values in settings.items():
        if section not in fields or not isinstance(values, dict):
            raise ValueError(f"Unknown settings table: {section}")
        for name, value in values.items():
            if name not in fields[section]:
                raise ValueError(f"Unknown setting: {section}.{name}")
            expected_type = fields[section][name]
            if (not isinstance(value, expected_type)
                    or (isinstance(value, bool) and expected_type is not bool)):
                raise ValueError(f"Invalid value type for {section}.{name}: {value!r}")
            defaults[name] = value

    if defaults.get("potential", "harmonic") not in POTENTIAL_LABELS:
        raise ValueError(f"simulation.potential must be one of: {', '.join(POTENTIAL_LABELS)}")
    if defaults.get("quantity", "density") not in ("density", "wavefunction"):
        raise ValueError('plot.quantity must be "density" or "wavefunction".')
    if defaults.get("zoom", "auto") not in ("auto", "full"):
        raise ValueError('plot.zoom must be "auto" or "full".')
    if defaults.get("density_scale", "linear") not in ("linear", "log"):
        raise ValueError('plot.density_scale must be "linear" or "log".')
    if defaults.get("hydrogen_orbital", "") not in ("", *ORBITALS):
        raise ValueError(f"Unknown hydrogen_orbital; choose from {', '.join(ORBITALS)}")
    if "output" in defaults:
        defaults["output"] = Path(defaults["output"]) if defaults["output"] else None
    return defaults


def parse_arguments(argv=None):
    # Read --config first, then use its values as defaults for the usual CLI.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path,
                               default=Path(__file__).with_name("settings.toml"))
    config_args, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--potential", choices=list(POTENTIAL_LABELS), default="harmonic")
    parser.add_argument("--n", type=int, default=25, help="Odd n gives an exact central cut")
    parser.add_argument("--length", type=float, default=8)
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--state", type=int, default=0,
                        help="Energy-sorted state index: 0 is the ground state")
    parser.add_argument("--quantity", choices=["density", "wavefunction"], default="density")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hydrogen-orbital", choices=["", *ORBITALS], default="")
    parser.add_argument("--zoom", choices=["auto", "full"], default="auto")
    parser.add_argument("--density-scale", choices=["linear", "log"], default="linear")
    parser.add_argument("--surface", action=argparse.BooleanOptionalAction, default=False)
    try:
        defaults = read_settings(config_args.config)
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read {config_args.config}: {error}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.n < 2:
        parser.error("--n / simulation.n must be at least 2.")
    if args.hydrogen_orbital and args.potential != "hydrogen":
        parser.error("hydrogen_orbital requires potential = 'hydrogen'.")
    if args.potential == "hydrogen" and args.n % 2:
        args.n += 1
        print(f"Hydrogen: using even n={args.n} to avoid sampling r=0.")
    if not np.isfinite(args.length) or args.length <= 0:
        parser.error("--length / simulation.length must be finite and positive.")
    if args.state < 0:
        parser.error("--state must be nonnegative.")
    if args.states < 1:
        parser.error("--states must be positive.")
    number_of_states = max(args.states, args.state + 1)
    if number_of_states >= args.n**3:
        parser.error("The requested state count must be smaller than n**3.")
    return args


def main():
    args = parse_arguments()
    number_of_states = max(args.states, args.state + 1)
    print(f"Settings: {args.config.resolve()}")
    selection_description = (f"orbital: {args.hydrogen_orbital}" if args.hydrogen_orbital
                             else f"state: {args.state}")
    print(f"Potential: {args.potential}; grid: {args.n}^3; "
          f"{selection_description}; plot: {args.quantity}")

    orbital_info = None
    if args.hydrogen_orbital:
        coordinates, spacing, energy, psi, orbital_info = solve_hydrogen_orbital(
            args.n, args.length, args.hydrogen_orbital)
        energies = np.array([energy])
        wavefunctions = psi.reshape(-1, 1)
        args.state = 0
    else:
        coordinates, spacing = make_grid(args.n, args.length)
        potential = make_potential(coordinates, args.potential, length=args.length)
        hamiltonian = build_hamiltonian(potential, spacing)
        energies, wavefunctions = solve_lowest_states(hamiltonian, number_of_states, spacing)
    filename = (
        f"{args.potential}_overview.png"
        if args.state == 0 and args.quantity == "density" else
        f"{args.potential}_state_{args.state}_{args.quantity}.png"
    )
    if orbital_info:
        filename = f"hydrogen_{args.hydrogen_orbital}_{args.quantity}.png"
    output = args.output or Path("plots") / filename
    if not output.is_absolute():
        output = args.config.resolve().parent / output
    plot_summary(coordinates, spacing, args.length, energies, wavefunctions,
                 args.potential, output, state=args.state, quantity=args.quantity,
                 zoom=args.zoom, density_scale=args.density_scale, orbital_info=orbital_info)
    if args.surface:
        psi = wavefunctions[:, args.state].reshape((len(coordinates),) * 3)
        surface_output = output.with_name(output.stem + "_3d.png")
        label = args.hydrogen_orbital or f"state {args.state}"
        plot_isosurfaces(coordinates, psi, f"{args.potential} | Numerical {label}", surface_output)
        print(f"3D surface: {surface_output}")
    if orbital_info:
        print("Method:", orbital_info["method"])
        print(f"Reference energy: {orbital_info['reference_energy']:.9f}; "
              f"relative difference: {orbital_info['relative_energy_error']:.3%}")
        print(f"Eigenpair residual: {orbital_info['relative_residual']:.3e}; "
              f"outer-shell probability: {orbital_info['outer_shell_probability']:.3%}")
    elif args.potential == "hydrogen" and energies[args.state] >= 0:
        print("WARNING: selected E>=0 is not a free-atom bound orbital. Increase the box.")
    print(f"Saved: {output.resolve()}")
    selected_name = args.hydrogen_orbital or f"State {args.state}"
    print(f"{selected_name} energy: {energies[args.state]:.9f}")
    print(f"Probability integral: {np.sum(abs(wavefunctions[:, args.state])**2) * spacing**3:.12f}")
    if not orbital_info:
        print(f"Lowest energies: {np.array2string(energies, precision=7)}")


if __name__ == "__main__":
    main()
