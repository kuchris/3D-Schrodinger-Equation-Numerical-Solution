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
import numpy as np

from schrodinger_3d import (
    build_hamiltonian,
    make_grid,
    make_potential,
    solve_lowest_states,
)


def reference_energies(kind, length, count):
    """Continuum reference levels, with degeneracies counted separately.

    Harmonic reference: infinite space, omega = 1.
    Box reference: Dirichlet cube of side length `length`.
    """
    if kind == "custom":
        return None  # Arbitrary user functions have no assumed exact spectrum.
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


def plot_summary(coordinates, spacing, length, energies, wavefunctions, kind, output,
                 state=0, quantity="density"):
    """Use one color scale for all three cuts of the selected state."""
    n = len(coordinates)
    psi = wavefunctions[:, state].reshape((n, n, n), order="C").copy()
    # The real eigenvector's overall sign is arbitrary; fix it for display.
    if psi.flat[np.argmax(np.abs(psi))] < 0:
        psi = -psi
    values = np.abs(psi)**2 if quantity == "density" else psi
    center = int(np.argmin(np.abs(coordinates)))
    cut_position = coordinates[center]

    # Add the known zero values at the box faces before drawing contours.
    plot_coordinates = np.r_[-length / 2, coordinates, length / 2]
    slices = [
        (values[:, :, center], "x", "y", f"XY plane | z = {cut_position:.3g}"),
        (values[:, center, :], "x", "z", f"XZ plane | y = {cut_position:.3g}"),
        (values[center, :, :], "y", "z", f"YZ plane | x = {cut_position:.3g}"),
    ]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12, 10), layout="constrained", facecolor="#faf9f6")
    grid = fig.add_gridspec(2, 2)
    density_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                    fig.add_subplot(grid[1, 0])]
    if quantity == "density":
        levels = np.linspace(0, values.max(), 35)
        cmap = "magma"
        color_label = rf"Density $|\psi_{{{state}}}|^2$"
    else:
        limit = np.max(np.abs(values))
        levels = np.linspace(-limit, limit, 41)
        cmap = "RdBu_r"
        color_label = rf"Wavefunction $\psi_{{{state}}}$ (real)"

    for ax, (plane, xlabel, ylabel, title) in zip(density_axes, slices):
        # Transpose: array's first coordinate belongs on the horizontal axis.
        contour = ax.contourf(
            plot_coordinates, plot_coordinates, np.pad(plane, 1).T,
            levels=levels, cmap=cmap,
        )
        if quantity == "wavefunction" and plane.min() < 0 < plane.max():
            ax.contour(plot_coordinates, plot_coordinates, np.pad(plane, 1).T,
                       levels=[0], colors="#555555", linewidths=0.6)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_aspect("equal")

    fig.colorbar(contour, ax=density_axes, shrink=0.75, label=color_label)

    ax = fig.add_subplot(grid[1, 1])
    index = np.arange(len(energies))
    reference = reference_energies(kind, length, len(energies))
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
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    annotation = f"State {state}: {energies[state]:.6f}"
    if reference is not None:
        error = abs(energies[state] - reference[state]) / abs(reference[state])
        annotation += f"\nReference: {reference[state]:.6f}\nRelative difference: {error:.3%}"
    else:
        annotation += "\nAnalytical reference: not supplied"
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

    name = {"harmonic": "3D harmonic oscillator", "box": "Particle in a 3D box",
            "custom": "Custom potential V(x, y, z)"}[kind]
    title = "Probability density" if quantity == "density" else "Signed wavefunction"
    fig.suptitle(f"{name} | State {state} | {title}", fontsize=17)
    interpretation = (
        "Slices show local density, not integrated probability. "
        if quantity == "density" else
        "Red / blue: positive / negative amplitude; white: zero. Overall sign is arbitrary. "
    )
    fig.supxlabel(
        f"hbar = mass = 1 | interior grid {n} x {n} x {n} | "
        f"L = {length:g} | h = {spacing:.4f} | zero boundary at the cube faces\n"
        + interpretation
        + {"harmonic": "Harmonic reference assumes infinite space.",
           "box": "Box reference uses the same cube boundaries.",
           "custom": "Custom potential; no analytical reference supplied."}[kind],
        fontsize=9,
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
        "plot": {"state": int, "quantity": str, "output": str},
    }
    defaults = {}
    for section, values in settings.items():
        if section not in fields or not isinstance(values, dict):
            raise ValueError(f"Unknown settings table: {section}")
        for name, value in values.items():
            if name not in fields[section]:
                raise ValueError(f"Unknown setting: {section}.{name}")
            if isinstance(value, bool) or not isinstance(value, fields[section][name]):
                raise ValueError(f"Invalid value type for {section}.{name}: {value!r}")
            defaults[name] = value

    if defaults.get("potential", "harmonic") not in ("harmonic", "box", "custom"):
        raise ValueError('simulation.potential must be "harmonic", "box", or "custom".')
    if defaults.get("quantity", "density") not in ("density", "wavefunction"):
        raise ValueError('plot.quantity must be "density" or "wavefunction".')
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
    parser.add_argument("--potential", choices=["box", "harmonic", "custom"], default="harmonic")
    parser.add_argument("--n", type=int, default=25, help="Odd n gives an exact central cut")
    parser.add_argument("--length", type=float, default=8)
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--state", type=int, default=0,
                        help="Energy-sorted state index: 0 is the ground state")
    parser.add_argument("--quantity", choices=["density", "wavefunction"], default="density")
    parser.add_argument("--output", type=Path, default=None)
    try:
        defaults = read_settings(config_args.config)
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read {config_args.config}: {error}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.n < 2:
        parser.error("--n / simulation.n must be at least 2.")
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
    print(f"Potential: {args.potential}; grid: {args.n}^3; "
          f"state: {args.state}; plot: {args.quantity}")

    coordinates, spacing = make_grid(args.n, args.length)
    potential = make_potential(coordinates, args.potential)
    hamiltonian = build_hamiltonian(potential, spacing)
    energies, wavefunctions = solve_lowest_states(hamiltonian, number_of_states, spacing)
    filename = (
        f"{args.potential}_overview.png"
        if args.state == 0 and args.quantity == "density" else
        f"{args.potential}_state_{args.state}_{args.quantity}.png"
    )
    output = args.output or Path("plots") / filename
    if not output.is_absolute():
        output = args.config.resolve().parent / output
    plot_summary(coordinates, spacing, args.length, energies, wavefunctions,
                 args.potential, output, state=args.state, quantity=args.quantity)
    print(f"Saved: {output.resolve()}")
    print(f"State {args.state} energy: {energies[args.state]:.9f}")
    print(f"Probability integral: {np.sum(abs(wavefunctions[:, args.state])**2) * spacing**3:.12f}")
    print(f"Lowest energies: {np.array2string(energies, precision=7)}")


if __name__ == "__main__":
    main()
