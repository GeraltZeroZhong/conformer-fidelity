# Development

Install the development dependencies from the repository root:

```bash
python -m pip install -e '.[evaluation,analysis,dev]'
```

## Source organization

Reusable code lives in `src/conformer_fidelity/`. The command-line interface dispatches to the same modules used by Python callers.

| Module | Responsibility |
| --- | --- |
| `chemistry` | Molecular identity, stereochemistry, atom ordering and symmetry |
| `generators` | Backend adapters and fixed-slot MMFF94s optimization |
| `selection` | Fixed-pool energy and diversity selection |
| `evaluation` | Qualification, aligned RMSD and pool-level metrics |
| `audits` | Upstream feature comparisons for stereoisomer pairs |
| `analysis` | Unit-level summaries and paired-contrast uncertainty |
| `reporting` | JSON and CSV exports |
| `artifacts`, `cli` | Serialization and command dispatch |

Backend dependencies are loaded when the corresponding operation runs. Keep basic imports and command help lightweight so Torch-based evaluation and JAX-based generation can use separate environments.

## Tests

```bash
python -m pytest
python -m ruff check src tests
```

Tests construct small molecules and arrays and mock external model execution. They run offline and exercise atom mapping, fixed-slot handling, qualification, selection and statistics. Add a focused test alongside a change to one of these behaviors.

For backend changes, test the adapter against the upstream revision you intend to support. Include that revision and the relevant environment details in the pull request.

## Build and install

```bash
python -m build
```

This creates a source archive and wheel in `dist/`. Install the wheel into a test environment, change to a directory outside the checkout, and run `conformer-fidelity --help` and the CPU demo. The demo requires the `evaluation` extra in that environment.

Keep examples focused on inputs users can supply. Generated arrays and reports belong in the output directories chosen for each calculation.
