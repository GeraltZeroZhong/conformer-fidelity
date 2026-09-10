"""Explicit local-checkpoint adapter for the official ET-Flow drugs-o3 system.

The adapter preserves the upstream featurization, ODE sampler and whole-graph
parity operation, and checks the atom-indexed graph against the input SMILES.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem, rdBase

from ._common import (
    UnsupportedGraph, coordinate_identity, fixed_slots, graph_record,
    input_molecule, input_to_native_mapping, validate_request,
)

def validate_etflow_graph(featurizer, native, graph):
    """Check actual native feature tensors and exported 2D atom-indexed chemistry."""
    import torch
    expected_edges = []
    for bond in native.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        expected_edges.extend(([a, b], [b, a]))
    if (graph.atomic_numbers.tolist() != [a.GetAtomicNum() for a in native.GetAtoms()]
            or graph.edge_index.T.tolist() != expected_edges):
        raise UnsupportedGraph('actual native model graph changed declared atom identity/order')
    expected_features = featurizer.get_atom_features_from_mol(native, True)
    expected_chiral = featurizer.get_chiral_centers_from_mol(native)
    if (not torch.equal(graph.node_attr, expected_features)
            or any(not torch.equal(graph[k], expected)
                   for k, expected in zip(('chiral_index', 'chiral_nbr_index', 'chiral_tag'), expected_chiral))):
        raise UnsupportedGraph('actual model/post-hoc features changed declared 2D chemistry')
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    exported = Chem.MolFromSmiles(graph.smiles, parser)
    labels = [a.GetAtomMapNum() for a in exported.GetAtoms()] if exported is not None else []
    if sorted(labels) != list(range(native.GetNumAtoms())):
        raise UnsupportedGraph('native exported atom-indexed graph lacks complete identity labels')
    exported = Chem.RenumberAtoms(exported, np.argsort(labels).tolist())
    for i, atom in enumerate(exported.GetAtoms()):
        atom.SetAtomMapNum(native.GetAtomWithIdx(i).GetAtomMapNum())
    mapping = input_to_native_mapping(native, exported)
    if not np.array_equal(mapping, np.arange(native.GetNumAtoms())):
        raise UnsupportedGraph('native exported atom labels do not preserve original indexed chemistry')


@contextmanager
def capture_native_batches(model, n_atoms, raw_batches, native_batches):
    """Tap the original parity operation; keep every completed batch on failure."""
    original = model.switch_parity_of_pos

    def recorded(pos, *args, **kwargs):
        raw_batches.append(pos.detach().cpu().numpy().reshape(-1, n_atoms, 3).copy())
        native = original(pos, *args, **kwargs)
        native_batches.append(native.detach().cpu().numpy().reshape(-1, n_atoms, 3).copy())
        return native

    model.switch_parity_of_pos = recorded
    try:
        yield
    finally:
        model.switch_parity_of_pos = original


class ETFlowGenerator:
    """Persistent official model; paths and execution device are caller supplied."""

    def __init__(self, *, source, checkpoint, device, config='drugs-o3',
                 batch_size=4, n_timesteps=50):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(n_timesteps, int) or n_timesteps < 1:
            raise ValueError('n_timesteps must be a positive integer')
        source, checkpoint = Path(source), Path(checkpoint)
        if not (source / 'etflow').is_dir():
            raise FileNotFoundError('source must be a local checkout containing etflow/')
        if not checkpoint.is_file():
            raise FileNotFoundError('an explicit local checkpoint is required')
        started = time.perf_counter()
        sys.path.insert(0, str(source.resolve()))
        try:
            import torch
            from etflow import BaseFlow
            from etflow.commons.configs import CONFIG_DICT
            from etflow.commons.featurization import MoleculeFeaturizer, get_mol_from_smiles
        except ImportError as exc:
            raise ImportError(
                'ET-Flow requires its upstream PyTorch/PyG dependencies. '
                'Install the official dependencies in the ET-Flow environment.'
            ) from exc
        self.torch, self.device = torch, str(device)
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'requested ET-Flow device is unavailable: {device}')
        self.featurizer = MoleculeFeaturizer()
        self.parse_native = get_mol_from_smiles
        if isinstance(config, dict):
            model_config = config
        elif str(config) in CONFIG_DICT:
            model_config = CONFIG_DICT[str(config)]().model_dict()
        else:
            model_config = json.loads(Path(config).read_text())
        self.model = BaseFlow.from_config(model_config)
        # As upstream: only load a checkpoint from a source the operator trusts.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['state_dict'] if 'state_dict' in payload else payload
        self.model.load_state_dict(state, strict=True)
        del payload, state
        self.model = self.model.to(self.device).eval()
        self.batch_size, self.n_timesteps = batch_size, n_timesteps
        if self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.metadata = dict(method='ET-Flow', checkpoint=str(checkpoint), source=str(source),
            config=model_config, device=self.device, execution_batch_size=batch_size,
            n_timesteps=n_timesteps, sampler_type='ode', rdkit=rdBase.rdkitVersion,
            torch=torch.__version__, posthoc='official whole-graph parity switch, unmodified',
            parameters=sum(p.numel() for p in self.model.parameters()),
            model_load_seconds=time.perf_counter()-started)

    def _synchronize(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def generate(self, smiles, *, count=20, seed=0):
        validate_request(count, seed)
        started = time.perf_counter()
        molecule = input_molecule(smiles)
        n_atoms = molecule.GetNumAtoms()
        result = dict(method=self.metadata, smiles=smiles, seed=seed, molecule=molecule,
            coordinates=np.empty((0, n_atoms, 3)), raw_coordinates=np.empty((0, n_atoms, 3)),
            rows=[], failure=None, target_access=False, external_optimization=False,
            external_selection=False, generation_seconds=0.)
        raw_batches, native_batches = [], []
        mapping = None
        try:
            native = self.parse_native(smiles)
            mapping = input_to_native_mapping(molecule, native)
            graph = self.featurizer.get_data_from_smiles(smiles)
            validate_etflow_graph(self.featurizer, native, graph)
            result.update(input_to_native=mapping.tolist(), native_graph=graph_record(native))
            self._synchronize()
            tick = time.perf_counter()
            try:
                with capture_native_batches(self.model, n_atoms, raw_batches, native_batches):
                    returned = self.model.predict([smiles], max_batch_size=self.batch_size,
                        num_samples=count, n_timesteps=self.n_timesteps, seed=seed,
                        device=self.device, sampler_type='ode', as_mol=False)[smiles]
                self._synchronize()
            finally:
                result['generation_seconds'] = time.perf_counter()-tick
            captured = np.concatenate(native_batches) if native_batches else np.empty((0, n_atoms, 3))
            if not np.array_equal(captured, np.asarray(returned, dtype=np.float64), equal_nan=True):
                raise ValueError('captured native batches differ from the unmodified predict return')
        except (RuntimeError, ValueError, KeyError) as exc:
            result['failure'] = f'{type(exc).__name__}: {exc}'
        if mapping is not None:
            if raw_batches:
                result['raw_coordinates'] = np.concatenate(raw_batches)[:, mapping]
            if native_batches:
                result['coordinates'] = np.concatenate(native_batches)[:, mapping]
                result['rows'] = coordinate_identity(molecule, result['coordinates'])
        result['graph_to_output_seconds'] = time.perf_counter()-started
        if self.device.startswith('cuda'):
            result['peak_allocated_bytes'] = self.torch.cuda.max_memory_allocated(self.device)
            result['peak_reserved_bytes'] = self.torch.cuda.max_memory_reserved(self.device)
        return fixed_slots(result, count)
