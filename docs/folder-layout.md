# Folder layout

| Path | Contents |
| --- | --- |
| `plot.cmd`, `direct.cmd`, `hydrogen.cmd` | Double-click launchers for Windows |
| `setup.cmd` | Creates `.venv` and installs `requirements.txt` |
| `config/` | The settings you edit: `settings.toml`, `direct.toml`, `hydrogen.toml` |
| `custom/potential.py` | Your own V(x, y, z) |
| `schrodinger3d/` | The solver; see the module table below |
| `examples/basic_usage.py` | A short, complete calculation in code |
| `tests/` | Automatic checks of the physics, settings, and pictures |
| `docs/` | These explanations and the README images |
| `outputs/plots/` | New pictures |
| `outputs/checks/` | Analysis reports (JSON) and other diagnostics |
| `outputs/archive/` | Older pictures kept from before the folders were reorganised |

`outputs/`, `.venv/`, and `.audit-venv/` stay on your computer only; Git
ignores them.

## Code modules

| Module | Job |
| --- | --- |
| `schrodinger3d/potentials.py` | Potential formulas and presets |
| `schrodinger3d/solver.py` | Grid, matrix H, and lowest eigenstates |
| `schrodinger3d/direct.py` | Full-grid solve and its report |
| `schrodinger3d/basis.py` | Sorts equal-energy states by mirror symmetry, after the solve |
| `schrodinger3d/angular.py` | Measures L² and Lz without changing the states |
| `schrodinger3d/hydrogen.py` | Named-orbital mode and its symmetry shortcut |
| `schrodinger3d/plotting.py` | Settings, command line, and 2D pictures |
| `schrodinger3d/display.py` | Zoom, centre cuts, and 3D surfaces |

Run the modules from the project folder, for example
`python -m schrodinger3d.plotting`. No installation is needed. The launchers
pass `-B` so that Python does not write `__pycache__` folders.

## History

The previous file layout is in commit `a8cc7f9`, and the original scripts are
on the `legacy` branch. When the outputs were reorganised, old files were moved,
not deleted. The local file `outputs/checks/organization-manifest.json` lists
each moved file with its old name, new location, size, and SHA-256 hash.
