# Folder layout and preserved local results

The root contains launchers, installation requirements, license, and README.
Simulation code lives in `schrodinger3d/`; no package installation is needed
when running its modules from the project root.

| Folder | Contents |
| --- | --- |
| `config/` | General and named-hydrogen TOML settings |
| `custom/` | The editable potential function |
| `schrodinger3d/` | Potential formulas, solver, hydrogen symmetry, and display |
| `examples/` | A short complete numerical calculation |
| `tests/` | Physical and display regression checks |
| `docs/` | Numerical method, paper sources, limits, and screenshots |
| `outputs/plots/` | Newly generated plots |
| `outputs/checks/` | Local diagnostics and organization manifest |
| `outputs/archive/previous_plots/` | Preserved plots from before organization |
| `outputs/archive/root_pycache/` | Preserved former root Python cache |

Existing local plot and diagnostic files were moved, not deleted. Their old
names, new locations, sizes, and SHA-256 hashes are recorded in the ignored local
`outputs/checks/organization-manifest.json`. The move verified all 26 files.
Older JSON/NPZ diagnostics are in `outputs/checks/`; prior PNGs are archived.

The prior tracked layout is recoverable from Git commit `a8cc7f9`. The original
historical scripts remain on `legacy`. Local `.venv` and `.audit-venv` were left
in place because moving a Python environment can break its executable paths.

Outputs, environments, and caches are ignored by Git. The launchers use `-B`
to avoid writing new bytecode caches during normal plot runs.
