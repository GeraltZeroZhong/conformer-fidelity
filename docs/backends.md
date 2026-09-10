# Backend setup

All generators accept a specified two-dimensional SMILES. The adapters preserve output-slot order, map coordinates to the input atom order, and record generation settings and native postprocessing. Reference coordinates enter the workflow at scoring.

Install each external backend in its upstream environment. Supply the source checkout, checkpoint and configuration paths shown below. Dependencies and assets are loaded when that backend runs.

## ETKDG and MMFF94s

The base NumPy/RDKit installation supports ETKDGv3 generation and MMFF94s optimization. Generation and optimization produce separate arrays and artifacts.

```python
from conformer_fidelity.generators.etkdg import generate, relax

original = generate("CCO", count=20, seed=7)
post = relax(
    "CCO", original["coordinates"],
    present_mask=original["present_mask"], max_iterations=200,
)
```

Use `select` to reduce a larger pool to an output budget using energy or geometric diversity. The [workflow guide](reproduction.md) shows these operations together.

## ET-Flow

Prepare the official ET-Flow checkout and its Torch/PyG environment.[^1] Set `--source` to the checkout root containing the `etflow/` package, and use the matching model checkpoint and configuration.

```bash
conformer-fidelity generate --method etflow --smiles 'CCO' --count 20 --seed 7 \
  --source /path/to/ETFlow --checkpoint /path/to/drugs-o3.ckpt \
  --config drugs-o3 --device cuda --output results/etflow.npz
```

The adapter defaults to `drugs-o3`, 50 sampling steps and batches of four. It uses the supplied model's feature construction, prediction path and whole-molecule parity correction.

## AvgFlow

Prepare the official AvgFlow JAX/Flax environment, the 52M reflow checkpoint and its generation configuration.[^2] Set `--source` to the checkout root containing `avgflow/`.

In an environment that already provides the package's NumPy/RDKit requirements, `python -m pip install -e . --no-deps` installs the project code while preserving the prepared dependency stack. Scoring can run separately in an environment with the `evaluation` extra.

```bash
conformer-fidelity generate --method avgflow --smiles 'CCO' --count 20 --seed 7 \
  --source /path/to/avgflow --checkpoint /path/to/avgflow_52m_reflow_ckpt.pkl \
  --config /path/to/avgflow_52m_reflow_gen.yaml --device cuda \
  --output results/avgflow.npz
```

The adapter uses EMA weights, two Euler steps on a uniform time grid, and the native mirror postprocessing. Its default execution batch contains four conformers; metadata records this separately from the upstream batch setting of 32. A partial final batch draws a full execution batch and retains the requested slots, with the total draw count recorded as `native_drawn_count`.

## CONFORGE

Install CDPKit/CONFORGE 1.3.0.[^3] If its Python modules are outside the environment's import path, supply their directory with `--source`:

```bash
conformer-fidelity generate --method conforge --smiles 'CCO' --count 20 --seed 0 \
  --source /path/to/cdpkit/python --timeout-ms 60000 \
  --output results/conforge.npz
```

The adapter uses `MEDIUM_SET_DIVERSE`, automatic sampling and the configured native timeout. This interface leaves random-seed control to CONFORGE: use `--seed 0`, and the metadata records `seed_controlled=false`. Short outputs and failures retain their requested slots.

## Source-conditioning audit

```bash
conformer-fidelity audit-conditioning --method avgflow --source /path/to/avgflow \
  --seed 0 --output results/conditioning.json
```

The audit executes upstream featurizer functions on three synthetic stereoisomer pairs and records tensor comparisons. It needs source code and its feature-construction dependencies, with SciPy for AvgFlow and Torch for ET-Flow. Model weights are unnecessary for this operation.

Feature equality describes the inspected featurizer under the supplied settings. Use `generate` and `identity` for the complementary measurement of stereochemistry in generated coordinates.

## Upstream sources

Record the source revision and checkpoint/configuration release used for a calculation. Backend code and assets retain the licenses supplied by their authors.

[^1]: ET-Flow official implementation. https://github.com/shenoynikhil/ETFlow
[^2]: AvgFlow official implementation. https://github.com/NVIDIA-BioNeMo/avgflow
[^3]: CDPKit official implementation and releases. https://github.com/molinfo-vienna/CDPKit
