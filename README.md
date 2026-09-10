# conformer-fidelity

Tools for checking whether molecular conformer ensembles preserve specified stereochemistry and recover reference conformations. Generate candidates, refine and select fixed-size pools, then measure stereochemical retention, geometric qualification and reference recovery.

The package provides a command-line interface and Python modules for ETKDG, CONFORGE, ET-Flow and AvgFlow workflows.

## Quick start

Use Python 3.10–3.12 in an active virtual or conda environment:

```bash
git clone https://github.com/GeraltZeroZhong/conformer-fidelity.git
cd conformer-fidelity
python -m pip install -e '.[evaluation,analysis]'
conformer-fidelity demo --output demo-output
```

The CPU demo generates ethanol conformers, applies MMFF94s optimization, and writes stereochemical diagnostics and recovery scores. It uses the first generated conformer as its reference, making the example self-contained. Choose a new output directory for each run.

The base installation, `python -m pip install -e .`, includes NumPy and RDKit. The `evaluation` extra adds the scoring dependencies; `analysis` adds SciPy. Neural backends use their upstream environments and locally supplied checkpoints; see [backend setup](docs/backends.md).

## Commands

| Command | Purpose |
| --- | --- |
| `generate` | Generate a conformer pool from a specified SMILES |
| `relax` | Apply MMFF94s optimization while retaining output slots |
| `select` | Select candidates using energy and geometric diversity |
| `identity` | Measure specified R/S and E/Z retention from coordinates |
| `score` | Measure pool recovery against a supplied reference |
| `audit-conditioning` | Compare upstream features for stereoisomer pairs |
| `statistics` | Summarize unit-level metrics and paired contrasts |
| `export` | Export metric and statistics tables to JSON/CSV |
| `demo` | Run the self-contained CPU example |

Use `conformer-fidelity <command> --help` for arguments. The same interface is available through `python -m conformer_fidelity`.

```bash
conformer-fidelity generate --method etkdg --smiles 'C[C@H](O)F' \
  --count 20 --seed 7 --output results/candidates.npz
conformer-fidelity relax --input results/candidates.npz \
  --iterations 200 --output results/postprocessed.npz
conformer-fidelity identity --input results/postprocessed.npz \
  --output results/identity.json
```

## Evaluation model

Coordinates retain the input graph's atom order. Generation and optimization preserve requested slots. Selection writes a fixed output budget and records each retained candidate's original pool index. Missing and failed outputs remain part of the requested budget.

Recovery is evaluated over three nested candidate sets: finite outputs, chemically eligible outputs, and jointly qualified outputs. A recovery hit requires the same conformer to pass the relevant checks and meet the aligned heavy-atom RMSD threshold. Independent stereochemical diagnostics identify errors even when another check fails first.

These metrics describe ligand-internal conformation and the configured geometric checks. See [formats and metrics](docs/formats.md) for atom mapping, qualification rules, denominators and uncertainty estimates.

## Documentation

- [Run a conformer workflow](docs/reproduction.md): generation, selection, scoring and analysis
- [Backend setup](docs/backends.md): environments, model assets and adapter settings
- [Formats and metrics](docs/formats.md): array schemas and measurement definitions
- [Development](docs/development.md): module organization, testing and packaging

## Development and license

```bash
python -m pip install -e '.[evaluation,analysis,dev]'
python -m pytest
python -m ruff check src tests
python -m build
```

Tests use synthetic molecules and arrays, with mocked external backends. Package code is available under the [MIT license](LICENSE). Upstream implementations and model assets retain their own licenses.
