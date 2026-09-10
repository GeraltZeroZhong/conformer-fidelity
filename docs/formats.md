# Formats and metrics

## Ensemble artifacts

Ensembles are stored as NPZ files with the following fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `coordinates` | `[slots, atoms, 3]` | Cartesian coordinates in Å |
| `present_mask` | `[slots]` | Boolean mask of returned outputs |
| `smiles` | Scalar string | Atom-ordered input SMILES |
| `metadata_json` | Optional scalar string | Settings and output diagnostics |

Generator outputs include explicit hydrogens in the order produced by `Chem.AddHs(Chem.MolFromSmiles(smiles))`. Some backends also retain pre-postprocessing coordinate arrays.

The presence mask records whether a slot was returned. Finiteness and qualification are measured separately. Missing slots contain NaNs, while returned nonfinite coordinates retain a true presence flag and fail qualification. All slots retain their place in the requested budget.

```python
from conformer_fidelity.artifacts import load_ensemble, save_ensemble

item = load_ensemble("results/candidates.npz")
save_ensemble("results/copy.npz", item["smiles"], item)
```

Writers require new output paths. JSON reports encode unavailable diagnostic values as `null`; NPZ preserves NaNs in coordinate arrays. NPZ loading uses `allow_pickle=False`.

## Reference coordinates

Scoring accepts one finite reference array with shape `[all_atoms, 3]` or `[heavy_atoms, 3]`. Both forms use the input graph's atom order; the heavy-only form follows its heavy atoms in increasing atom-index order. Use this form for experimental references with unobserved hydrogens.

For the CLI, wrap the reference in a one-slot ensemble with the same `smiles` string as the candidate artifact. For example, given an already mapped heavy-atom NPY array:

```python
import numpy as np
from conformer_fidelity.artifacts import save_ensemble

reference = np.load("inputs/reference-heavy.npy", allow_pickle=False)
save_ensemble("inputs/reference.npz", "CCCO", {
    "coordinates": reference[None, :, :],
    "present_mask": np.array([True]),
})
```

```bash
conformer-fidelity score --input results/selected.npz --reference inputs/reference.npz \
  --reference-index 0 --requested 20 --output results/scores.json
```

Atom correspondence must be established when preparing the reference. This includes extraction and mapping of ligands from PDB or SDF files. Matching atom counts or canonical molecular identities alone leaves coordinate ordering unresolved; the CLI requires the same atom-ordered SMILES in both artifacts.

The Python scoring interface accepts arrays directly:

```python
from conformer_fidelity.evaluation import score_pool

result = score_pool(
    smiles, candidates, reference_heavy_coordinates,
    present_mask=present, denominator=20,
)
```

## Optimization and selection

`relax` applies MMFF94s optimization to the input slots and writes a separate ensemble. Status `0` denotes convergence; status `1` retains the finite coordinates at the iteration limit. Setup or optimization failures produce an unavailable output in the same slot, with the failure recorded.

Selection uses the supplied pool, chemical eligibility, MMFF94s single-point energies and symmetry-aware pairwise heavy-atom RMSD. Candidate coordinates remain unchanged. The supported output size is 0–20.

| Rule | Selection criterion |
| --- | --- |
| `energy` | Lowest energy first |
| `facility` | Largest mean reduction in distance to selected representatives |
| `energy_facility` | Equal-weight combination of energy and facility-gain ranks |
| `kcenter` | Farthest from selected candidates, starting at the lowest-energy candidate |
| `energy_rmsd05` | Energy order with separation greater than 0.5 Å |
| `energy_rmsd10` | Energy order with separation greater than 1.0 Å |

The CLI records original pool indices and pads unfilled output slots with absent/NaN entries. In Python, `select_indices` returns an index array, and `select_conformers` returns a dictionary containing `indices` and selection diagnostics. Both return the retained indices without padding. The diversity-threshold rules may retain fewer candidates than the output budget.

## Qualification and recovery

The three nested output sets are:

- F: finite returned conformers
- O: F conformers passing bond-distance, specified-stereochemistry and bridge-torsion eligibility checks
- Q: O conformers also passing six configured PoseBusters geometric tests

The geometric tests cover bond and angle-associated distance bounds, internal clashes, aromatic-ring flatness, non-aromatic six-membered-ring non-flatness and double-bond flatness. They use the distance-geometry and flatness modules from PoseBusters 0.6.5. Q describes this ligand-only geometric subset; energy sampling and receptor-contact analysis are separate measurements.

RMSD uses heavy atoms, translation, proper rotation and stereochemistry-preserving graph symmetries. A hit requires the same candidate to belong to the relevant set and fall within the distance threshold.

| Fields | Meaning |
| --- | --- |
| `finite_hit_1`, `finite_hit_2` | At least one F candidate within 1 Å or 2 Å |
| `original_hit_1`, `original_hit_2` | Corresponding O-set hits |
| `qhit_1`, `qhit_2` | Corresponding Q-set hits |
| `original_hit_loss_1`, `original_hit_loss_2` | F hit minus O hit |
| `geometry_hit_loss_1`, `geometry_hit_loss_2` | O hit minus Q hit |
| `finite_best_rmsd`, `original_best_rmsd`, `qualified_best_rmsd` | Minimum in-set RMSD; 100 Å for an empty set |
| `output_fraction` | Present outputs / requested slots |
| `eligible_fraction` | O outputs / requested slots |
| `geometry_fraction` | Q outputs / requested slots |
| `qualified_output_fraction` | Q outputs / present outputs; zero when none are present |

The 100 Å value is an empty-set penalty. Report empty-set counts alongside averages that include this value. With 20 requested slots and 13 returned outputs that all qualify, `geometry_fraction` is `13/20` and `qualified_output_fraction` is `13/13`.

`identity` measures specified R/S and E/Z states independently of the sequential eligibility checks. Molecules without specified stereochemistry have `specified_stereo_retained=null`, indicating that the input supplies no stereochemical state to test.

## Unit-level statistics

The analysis NPZ contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `means` | `[units, methods, metrics]` | Finite measurements averaged within each independent unit |
| `names` | `[methods]` | Method labels |
| `fields` | `[metrics]` | Metric labels |
| `strata` | Optional `[units]` | Stratum labels; defaults to one group |

Resolve failed runs according to the calculation's failure rule before aggregation. Repeats contribute to their unit's mean. Each stratum needs at least two independent units. Strata receive equal total weight by default; use `stratum_proportions` to specify alternative population weights.

For paired contrasts, declare the method/metric pair and its known mathematical range:

```python
from conformer_fidelity.analysis import Contrast, run_statistics

run_statistics(
    "inputs/per-unit.npz", "results/statistics",
    contrasts=[Contrast(
        name="post-minus-native",
        left_method="post", left_metric="qhit_1",
        right_method="native", right_metric="qhit_1", domain=(-1.0, 1.0),
    )],
    metric_domains={"qhit_1": (0.0, 1.0)},
    draws=50000, seed=0,
)
```

Hoeffding intervals use independent bounded units and the declared contrast domains. Bootstrap intervals and optional cluster-based intervals provide additional views of uncertainty under their respective resampling and dependence assumptions. Specify outcome domains from their mathematical ranges, independently of the observed extrema.

The CLI accepts these declarations through `--spec`; see [the contrast example](../examples/contrasts.json) and [analysis workflow](reproduction.md#analyze-paired-metrics). With no contrasts, it produces descriptive summaries.
