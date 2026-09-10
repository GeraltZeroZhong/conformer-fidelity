"""SMILES-based adapter for the official AvgFlow 52M reflow model."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem, rdBase

from ._common import fixed_slots, validate_request


@contextmanager
def numpy_feature_seed(seed, *, eigensolver_module=None):
    """Condition-local official PE signs and explicit sparse-eigensolver start.

    SciPy otherwise asks ARPACK to choose its own starting vector for n>=100.
    Supplying v0 controls this randomness while preserving the matrix, solver,
    number of eigenvectors and sign distribution. NumPy and eigensolver state
    are restored after the serial featurization call.
    """
    previous = np.random.get_state()
    np.random.seed(seed)
    original_eigs = eigensolver_module.eigs if eigensolver_module is not None else None
    if eigensolver_module is not None:
        def seeded_eigs(matrix, *args, **kwargs):
            kwargs.setdefault('v0', np.random.Generator(np.random.PCG64(seed)).uniform(-1., 1., matrix.shape[0]))
            return original_eigs(matrix, *args, **kwargs)
        eigensolver_module.eigs = seeded_eigs
    try:
        yield
    finally:
        np.random.set_state(previous)
        if eigensolver_module is not None:
            eigensolver_module.eigs = original_eigs


def input_graph(smiles, max_n=200):
    if not isinstance(smiles, str) or not smiles or '|' in smiles or any(x.isspace() for x in smiles):
        raise ValueError('a plain 2D SMILES is required; no CXSMILES or coordinates')
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('invalid or disconnected input graph')
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError('radicals are outside the current common-evaluator scope')
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumConformers():
        raise ValueError('input unexpectedly contains coordinates')
    if molecule.GetNumAtoms() > max_n:
        raise ValueError(f'graph exceeds native max_n={max_n} including explicit H')
    return molecule


def stereo_signature(molecule, positions=None):
    from conformer_fidelity.chemistry.stereo import stereochemistry_signature
    return stereochemistry_signature(molecule, positions=positions, from_geometry=positions is not None)


def original_stereo_retained(reference, candidate):
    from conformer_fidelity.chemistry.stereo import stereochemistry_retention
    return stereochemistry_retention(reference, candidate).all_retained


def native_postprocess(molecule, coordinates, rdkit_helpers):
    """Apply the upstream whole-molecule mirror using the input graph's stereo."""
    canonical_input = Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)
    reference = stereo_signature(molecule)
    output, rows = [], []
    for i, raw in enumerate(np.asarray(coordinates, dtype=np.float64)):
        finite = bool(np.isfinite(raw).all())
        xyz = raw.copy()
        flipped, retained = False, False
        if finite:
            native_mol = rdkit_helpers.add_conformer_to_mol(molecule, raw)
            post = rdkit_helpers.post_hoc_flip_mol(gt_smis=canonical_input, gen_mol=native_mol)
            xyz = post.GetConformer().GetPositions()
            if np.array_equal(xyz, raw):
                flipped = False
            elif np.array_equal(xyz, -raw):
                flipped = True
            else:
                raise ValueError('upstream post-hoc helper made an operation other than the native whole-molecule mirror')
            retained = original_stereo_retained(reference, stereo_signature(molecule, xyz))
        output.append(xyz)
        rows.append(dict(native_index=i, finite=finite, native_posthoc_flip=flipped,
            specified_stereo_retained=retained, identity_passed=finite and retained,
            failure='nonfinite_coordinates' if not finite else 'specified_stereochemistry_changed' if not retained else None))
    shape = (-1, molecule.GetNumAtoms(), 3)
    return np.asarray(output, dtype=np.float64).reshape(shape), rows, canonical_input


class AvgFlowGenerator:
    """Persistent official 52M reflow/EMA/two-Euler generator.

    source is a local checkout containing avgflow/. checkpoint and YAML config
    are local files. Batch rounding retains the upstream random-key sequence and
    reports native_drawn_count. Featurization uses process-local NumPy/ARPACK
    state; run concurrent generators in separate processes.
    """

    def __init__(self, *, source, checkpoint, config, device, batch_size=4):
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError('batch_size must be within the native 1..32 range')
        source, checkpoint, config_path = Path(source), Path(checkpoint), Path(config)
        if not (source / 'avgflow').is_dir():
            raise FileNotFoundError('source must be a local checkout containing avgflow/')
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError('explicit local checkpoint and YAML config are required')
        started = time.perf_counter()
        sys.path.insert(0, str(source.resolve()))
        try:
            import jax
            import yaml
            from flax import nnx
            from avgflow.nn.model import ConformerFlowTransformer
            from avgflow.utils.model_utils import load_ckpt_from_pkl
            from avgflow.flow_matcher.flow_matcher import _generate_ode
            from avgflow.dataloader.jnp_dataloader import SingleGenerationLoader
            from avgflow.dataloader.preprocess import mol2features
        except ImportError as exc:
            raise ImportError(
                'AvgFlow requires its upstream JAX/Flax/Diffrax environment. '
                'Install the official dependencies in the AvgFlow environment.'
            ) from exc
        platform, _, index = str(device).partition(':')
        if platform == 'cuda':
            platform = 'gpu'
        if platform not in ('cpu', 'gpu'):
            raise ValueError('device must be cpu, gpu[:index] or cuda[:index]')
        try:
            self.device = jax.devices(platform)[int(index or 0)]
        except (RuntimeError, IndexError, ValueError) as exc:
            raise RuntimeError(f'requested AvgFlow device is unavailable: {device}') from exc
        configuration = yaml.safe_load(config_path.read_text())
        if not (configuration['use_ema'] and configuration['n_steps'] == 2
                and configuration['solver'] == 'euler'
                and configuration['t_schedule'] == 'uniform'
                and configuration['model']['max_n'] == 200):
            raise ValueError('this adapter requires the official 52M reflow/EMA/two-Euler configuration')
        self.jax, self.batch_size = jax, batch_size
        self.featurize, self.loader_type, self.sample = mol2features, SingleGenerationLoader, _generate_ode
        self.feature_module = importlib.import_module('avgflow.dataloader.data_utils')
        self.rdkit_helpers = importlib.import_module('avgflow.utils.rdkit_utils')
        model_config = configuration['model']
        with jax.default_device(self.device):
            model = ConformerFlowTransformer(max_n=model_config['max_n'], node_raw_feat_dim=model_config['feat_dim'],
                pe_dim=model_config['pe_dim'], graph_feat_dim=model_config['graph_feat_dim'],
                cond_dim=model_config['cond_dim'], n_edge_type=model_config['n_edge_type'],
                edge_type_emb_dim=model_config['edge_type_emb_dim'], node_dim=model_config['node_dim'],
                n_heads=model_config['n_heads'], n_layers=model_config['n_layers'],
                n_registers=model_config['n_registers'], rngs=nnx.Rngs(0))
            self.model = load_ckpt_from_pkl(str(checkpoint), model, use_ema=True)
            for value in jax.tree.leaves(nnx.state(self.model)):
                if hasattr(value, 'block_until_ready'):
                    value.block_until_ready()
        self.compiled = False
        self.config = configuration
        self.metadata = dict(method='AvgFlow52M-reflow-EMA-2Euler', checkpoint=str(checkpoint),
            native_model=model_config, native_batch_size=32, execution_batch_size=batch_size,
            max_n=200, use_ema=True, num_steps=2, solver='euler', t_schedule='uniform',
            featurize_mode='drugs', rdkit=rdBase.rdkitVersion, jax=jax.__version__,
            device=str(self.device), upstream=str(source), config_path=str(config_path),
            no_training_optimizer_created=True, persistent_model=True,
            feature_randomness='condition-local NumPy seed for official PE signs; independent PCG64(seed) uniform(-1,1) ARPACK v0',
            posthoc='authors whole-molecule mirror against canonical input graph SMILES',
            model_load_seconds=time.perf_counter() - started)

    def generate(self, smiles, *, count=20, seed=0):
        validate_request(count, seed)
        with self.jax.default_device(self.device):
            return self._generate(smiles, count=count, seed=seed)

    def _generate(self, smiles, *, count, seed):
        started = time.perf_counter()
        result = dict(smiles=smiles, seed=seed, feature_seed=seed, requested_outputs=count,
            method=self.metadata, molecule=None, raw_coordinates=np.empty((0, 0, 3)),
            coordinates=np.empty((0, 0, 3)), rows=[], failure=None, target_access=False,
            external_optimization=False, external_selection=False, output_count=0,
            generation_seconds=0., feature_seconds=0., first_jit_call_included=False,
            first_compile_plus_execution_seconds=0., native_drawn_count=0)
        generated = []
        try:
            molecule = input_graph(smiles)
            result['molecule'] = molecule
            result['coordinates'] = np.empty((0, molecule.GetNumAtoms(), 3))
            result['raw_coordinates'] = np.empty_like(result['coordinates'])
            tick = time.perf_counter()
            with numpy_feature_seed(seed, eigensolver_module=self.feature_module):
                graph = self.featurize(molecule, 'drugs')
            if graph['node_feats'].shape[0] != molecule.GetNumAtoms():
                raise ValueError('native featurizer changed atom count')
            batch = self.loader_type(graph, max_n=200, batchsize=self.batch_size).get_batch()
            result['feature_seconds'] = time.perf_counter() - tick
            key = self.jax.random.key(seed)
            key, sample_key = self.jax.random.split(key)
            for _ in range((count + self.batch_size - 1) // self.batch_size):
                tick = time.perf_counter()
                trajectory = self.sample(self.model, batch, sample_key, 2, 'uniform', 2., 'euler', 1e-5, 1e-4)
                trajectory.block_until_ready()
                raw = np.asarray(trajectory[-1, :, :molecule.GetNumAtoms(), :], dtype=np.float64)
                seconds = time.perf_counter() - tick
                result['generation_seconds'] += seconds
                if not self.compiled:
                    result['first_jit_call_included'] = True
                    result['first_compile_plus_execution_seconds'] = seconds
                    self.compiled = True
                generated.extend(raw[:count - len(generated)])
                result['native_drawn_count'] += self.batch_size
                key, sample_key = self.jax.random.split(key)
            result['raw_coordinates'] = np.asarray(generated, dtype=np.float64)
            xyz, rows, canonical = native_postprocess(molecule, result['raw_coordinates'], self.rdkit_helpers)
            result.update(coordinates=xyz, rows=rows, input_canonical_smiles_no_h=canonical,
                native_to_rdkit=list(range(molecule.GetNumAtoms())),
                original_atom_maps=[a.GetAtomMapNum() for a in molecule.GetAtoms()])
        except (RuntimeError, ValueError, KeyError) as exc:
            result['failure'] = type(exc).__name__ + ': ' + str(exc)
            # Retain every completed raw batch for diagnosis; never regenerate.
            if generated:
                result['raw_coordinates'] = np.asarray(generated, dtype=np.float64)
        result['output_count'] = len(result['coordinates'])
        result['missing_output_slots'] = count - result['output_count']
        result['identity_passed_count'] = sum(r['identity_passed'] for r in result['rows'])
        result['graph_to_output_seconds'] = time.perf_counter() - started
        return fixed_slots(result, count)
