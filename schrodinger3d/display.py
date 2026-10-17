"""Display numerical wavefunctions without changing their computed values."""

from pathlib import Path
import numpy as np


def central_wavefunction_slices(psi, coordinates):
    """Interpolate psi to zero on each axis, then let the caller square it.

    Averaging densities instead would fill in an odd orbital's nodal plane.
    Returns XY, XZ, YZ arrays with their coordinate labels.
    """
    upper = int(np.searchsorted(coordinates, 0))
    slices = []
    for axis, horizontal, vertical in [(2, "x", "y"), (1, "x", "z"), (0, "y", "z")]:
        if abs(coordinates[upper]) < 1e-12:
            plane = np.take(psi, upper, axis=axis)
        else:
            lower = upper - 1
            weight = -coordinates[lower] / (coordinates[upper] - coordinates[lower])
            plane = ((1 - weight) * np.take(psi, lower, axis=axis)
                     + weight * np.take(psi, upper, axis=axis))
        slices.append((plane, horizontal, vertical))
    return slices


def probability_view_limit(coordinates, density, fraction=0.995):
    """Find a centered cube retaining at least `fraction` of sampled probability."""
    total = density.sum()
    order = np.argsort(abs(coordinates))
    # Bound each axis's omitted mass by a third of the total allowance.
    target = 1 - (1 - fraction) / 3
    limits = []
    for axis in range(3):
        marginal = density.sum(axis=tuple(i for i in range(3) if i != axis))
        cumulative = np.cumsum(marginal[order]) / total
        index = min(int(np.searchsorted(cumulative, target)), len(order) - 1)
        limits.append(abs(coordinates[order[index]]))
    spacing = coordinates[1] - coordinates[0]
    full_limit = abs(coordinates[0]) + spacing
    return min(full_limit, max(limits) + spacing)


def plot_isosurfaces(coordinates, psi, title, output, fraction=0.04):
    """Draw +/-psi surfaces where density equals `fraction` of its peak.

    Red/blue are amplitude signs, not electron trajectories or sharp edges.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    spacing = coordinates[1] - coordinates[0]
    amplitude = np.sqrt(fraction) * np.max(abs(psi))
    field = np.pad(psi, 1).astype(np.float32)
    origin = coordinates[0] - spacing
    fig = plt.figure(figsize=(9, 8), layout="constrained", facecolor="#faf9f6")
    ax = fig.add_subplot(projection="3d")
    extent = 0
    for sign, color in [(1, "#d14b43"), (-1, "#367abb")]:
        level = sign * amplitude
        if not field.min() < level < field.max():
            continue
        vertices, faces, _, _ = marching_cubes(field, level=level, spacing=(spacing,) * 3)
        vertices += origin
        extent = max(extent, float(abs(vertices).max()))
        mesh = Poly3DCollection(vertices[faces], facecolor=color, edgecolor="none", alpha=0.8)
        ax.add_collection3d(mesh)
    limit = extent * 1.15
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
           xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    ax.legend(handles=[Patch(color="#d14b43", label="Positive amplitude"),
                       Patch(color="#367abb", label="Negative amplitude")], loc="upper left")
    fig.suptitle(title, fontsize=16)
    fig.supxlabel(f"Computed finite-difference wavefunction | density isosurface = {fraction:.0%} of peak\n"
                  "The surface threshold controls visibility; it is not an orbital boundary.", fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
