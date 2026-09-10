# Run a conformer workflow

This guide composes the package commands into a calculation with your own molecules and references. For a self-contained installation check, start with the [CPU demo](../README.md#quick-start).

## Generate, optimize and select

The following example generates 80 propanol conformers, applies fixed-step MMFF94s optimization, and selects up to 20 representatives:

```bash
conformer-fidelity generate --method etkdg --smiles 'CCCO' --count 80 --seed 7 \
  --output results/pool.npz
conformer-fidelity relax --input results/pool.npz --iterations 200 \
  --output results/relaxed.npz
conformer-fidelity select --input results/relaxed.npz --method facility --count 20 \
  --output results/selected.npz
```

Each operation writes a new artifact. Selection records the retained candidates' original pool indices, preserving the connection to the generated structures. Unfilled output slots remain absent. Use a fresh output path when repeating a calculation.

For neural generation or CONFORGE, replace the `generate` command with the corresponding [backend invocation](backends.md).

## Measure identity and recovery

```bash
conformer-fidelity identity --input results/selected.npz \
  --output results/identity.json
conformer-fidelity score --input results/selected.npz --reference inputs/reference.npz \
  --requested 20 --output results/scores.json
```

`identity` reads specified stereochemistry independently from each returned structure. `score` additionally uses a reference conformation of the same molecule. Prepare `inputs/reference.npz` with the same SMILES and explicit atom correspondence, following the [reference format](formats.md#reference-coordinates). Experimental heavy-atom coordinates can be supplied directly in the input graph's heavy-atom order.

The scores retain the requested denominator and report finite, chemically eligible and jointly qualified recovery. The reference is used only by `score` in this workflow.

## Analyze paired metrics

Average repeated measurements within each independent unit, then assemble a unit × method × metric NPZ cube as described in [formats and metrics](formats.md#unit-level-statistics).

```bash
conformer-fidelity statistics --input inputs/per-unit.npz \
  --spec examples/contrasts.json --output results/statistics --draws 50000 --seed 0
conformer-fidelity export --input inputs/per-unit.npz \
  --statistics results/statistics/statistics.json --output results/tables
```

The [contrast example](../examples/contrasts.json) defines `post-minus-native` for `qhit_1`. Match its method and metric names to your cube. Omitting `--spec` produces descriptive summaries. Choose contrast domains and stratum weights from the mathematical ranges and sampling design before interpreting the resulting intervals.

## Record a calculation

Retain the ordered molecular inputs, reference mappings, package versions, backend source revision, model/configuration release, seeds, requested counts, and optimization and selection settings. Adapter metadata records the settings available during execution, including batch behavior and native seed control. Together with the input files, these records define the calculation to reproduce.
