"""Choose a basis inside each degenerate energy level AFTER the full solve.

H alone fixes the energies and eigenspaces, not a basis inside a degenerate
level. This module picks one using operations that leave the sampled potential
unchanged: the reflections x->-x, y->-y, z->-z, then the x<->y swap. Each
operation is tested on V first, so it commutes with the discrete H; a potential
without that symmetry is left alone. Vectors are only recombined within one
level, so they remain eigenvectors of the same H.

There is no orbital table, label input, template, or spherical harmonic.
The x<->y swap makes z the special axis, the same convention as textbook
real orbitals.
"""

import numpy as np


OPERATIONS = {
    "Rx": lambda values: np.flip(values, axis=0),
    "Ry": lambda values: np.flip(values, axis=1),
    "Rz": lambda values: np.flip(values, axis=2),
    "Pxy": lambda values: values.swapaxes(0, 1),
}


def potential_symmetries(potential, rtol=1e-10):
    """Return the names of OPERATIONS that leave the sampled potential unchanged.

    The kinetic term on the centered cubic grid is invariant under all of them,
    so an operation that preserves V commutes with the whole discrete H.
    """
    scale = max(1.0, float(np.max(np.abs(potential))))
    return [name for name, operation in OPERATIONS.items()
            if np.max(np.abs(operation(potential) - potential)) <= rtol * scale]


def degenerate_levels(energies, tolerance=1e-8):
    """Group sorted energies that agree within tolerance * max(1, |E|)."""
    levels = [[0]]
    for index in range(1, len(energies)):
        first = energies[levels[-1][0]]
        if abs(energies[index] - first) <= tolerance * max(1.0, abs(first)):
            levels[-1].append(index)
        else:
            levels.append([index])
    return levels


def _restricted(vectors, shape, operation):
    """Matrix of `operation` in an orthonormal column basis, and its leakage.

    Leakage is the part of operation(vectors) outside their span; it is ~0
    only when the span is invariant under the operation.
    """
    images = np.column_stack([operation(column.reshape(shape)).ravel()
                              for column in vectors.T])
    matrix = vectors.T @ images
    matrix = (matrix + matrix.T) / 2
    leakage = np.linalg.norm(images - vectors @ matrix) / np.sqrt(vectors.shape[1])
    return matrix, leakage


def symmetry_adapted_states(states, shape, energies, symmetries,
                            tolerance=1e-8, invariance_tolerance=1e-6):
    """Recombine states within each degenerate level; return states and notes.

    states: (n**3, count) columns with equal norms, as from solve_lowest_states.
    A level whose span is not closed under a symmetry (for example a multiplet
    cut by the requested state count) is left unrotated and reported.
    Within a level, columns are ordered by their operation eigenvalues, -1 first.
    """
    rotated = states.copy()
    notes = []
    for level in degenerate_levels(energies, tolerance):
        if len(level) == 1:
            continue
        scale = np.linalg.norm(states[:, level[0]])
        vectors = states[:, level] / scale
        open_under = [name for name in symmetries
                      if _restricted(vectors, shape, OPERATIONS[name])[1] > invariance_tolerance]
        if open_under:
            notes.append(f"States {level[0]}-{level[-1]} left unrotated: their span is not "
                         f"closed under {', '.join(open_under)}; the level may be cut by "
                         "the requested state count.")
            continue

        # Split the level into blocks with definite eigenvalues, one operation at a time.
        blocks = [vectors]
        for name in symmetries:
            refined = []
            for block in blocks:
                matrix, leakage = _restricted(block, shape, OPERATIONS[name])
                if block.shape[1] == 1 or leakage > invariance_tolerance:
                    refined.append(block)  # Already definite, or this operation mixes blocks.
                    continue
                eigenvalues, rotation = np.linalg.eigh(matrix)
                block = block @ rotation
                negative = eigenvalues < 0
                refined.extend(part for part in (block[:, negative], block[:, ~negative])
                               if part.shape[1])
            blocks = refined
        unresolved = [block.shape[1] for block in blocks if block.shape[1] > 1]
        if unresolved:
            notes.append(f"States {level[0]}-{level[-1]}: a {max(unresolved)}-dimensional "
                         "part is not separated by the available symmetries.")
        basis = np.column_stack(blocks)
        # The overall sign is arbitrary; make each largest component positive.
        signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(basis.shape[1])])
        rotated[:, level] = basis * signs * scale
    return rotated, notes


def symmetry_expectations(psi, symmetries):
    """<psi|O|psi>/<psi|psi> for each operation; +/-1 means a definite eigenvalue."""
    norm_squared = float(np.vdot(psi, psi).real)
    return {name: float(np.vdot(psi, OPERATIONS[name](psi)).real / norm_squared)
            for name in symmetries}
