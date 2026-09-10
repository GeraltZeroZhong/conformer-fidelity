"""Native adapter contracts with mocks: no checkpoints, GPUs or native searches."""
from contextlib import nullcontext
import builtins
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from conformer_fidelity.generators import avgflow, conforge, etflow
from conformer_fidelity.generators._common import input_molecule


def mirror_helpers(flip=False):
    def add(molecule, coordinates):
        copied = Chem.Mol(molecule)
        copied.RemoveAllConformers()
        conf = Chem.Conformer(copied.GetNumAtoms())
        conf.Set3D(True)
        for i, xyz in enumerate(coordinates):
            conf.SetAtomPosition(i, tuple(map(float, xyz)))
        copied.AddConformer(conf)
        return copied

    def post(*, gt_smis, gen_mol):
        assert gt_smis == Chem.MolToSmiles(Chem.RemoveHs(gen_mol), canonical=True)
        if flip:
            conf = gen_mol.GetConformer()
            for i, xyz in enumerate(conf.GetPositions()):
                conf.SetAtomPosition(i, tuple(-xyz))
        return gen_mol

    return SimpleNamespace(add_conformer_to_mol=add, post_hoc_flip_mol=post)


def test_avg_feature_seed_and_sparse_start_restore_upstream_state():
    np.random.seed(987)
    before = np.random.get_state()
    calls = []

    def eigs(matrix, **kwargs):
        calls.append(kwargs)
        return kwargs['v0']

    module = SimpleNamespace(eigs=eigs)
    with avgflow.numpy_feature_seed(101, eigensolver_module=module):
        first = module.eigs(np.eye(104), k=33, which='SR')
        signs = np.random.randint(0, 2, 32)
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert module.eigs is eigs
    with avgflow.numpy_feature_seed(101, eigensolver_module=module):
        np.testing.assert_array_equal(first, module.eigs(np.eye(104), k=33, which='SR'))
        np.testing.assert_array_equal(signs, np.random.randint(0, 2, 32))
    assert calls[0]['k'] == 33 and calls[0]['which'] == 'SR'


def test_avg_native_mirror_uses_canonical_graph_and_does_not_mutate_raw():
    molecule = avgflow.input_graph('OC(CC)C')
    raw = np.arange(molecule.GetNumAtoms()*3, dtype=float).reshape(1, -1, 3)
    original = raw.copy()
    output, rows, canonical = avgflow.native_postprocess(molecule, raw, mirror_helpers(True))
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(output, -raw)
    assert rows[0]['native_posthoc_flip']
    assert canonical == Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True)


def test_avg_nonfinite_slot_is_not_postprocessed_or_refilled():
    molecule = avgflow.input_graph('CC')
    raw = np.full((1, molecule.GetNumAtoms(), 3), np.nan)
    xyz, rows, _ = avgflow.native_postprocess(molecule, raw, SimpleNamespace())
    assert xyz.shape == raw.shape and np.isnan(xyz).all()
    assert rows[0]['failure'] == 'nonfinite_coordinates'


def mock_avg_generator(*, fail_on_batch=None):
    generator = avgflow.AvgFlowGenerator.__new__(avgflow.AvgFlowGenerator)
    generator.device, generator.batch_size, generator.compiled = 'cpu', 4, False
    generator.metadata, generator.model = {}, object()
    generator.feature_module = SimpleNamespace(eigs=lambda *a, **k: None)
    generator.featurize = lambda molecule, dataset: {
        'node_feats': np.ones((molecule.GetNumAtoms(), 75))}
    generator.loader_type = lambda graph, **kwargs: SimpleNamespace(get_batch=lambda: graph)
    generator.jax = SimpleNamespace(default_device=lambda device: nullcontext(),
        random=SimpleNamespace(key=lambda seed: seed, split=lambda seed: (seed+1, seed+2)))
    generator.rdkit_helpers = mirror_helpers()
    calls = []

    class Trajectory:
        def __init__(self, value):
            self.values = np.full((3, 4, 200, 3), float(value))

        def block_until_ready(self):
            pass

        def __getitem__(self, index):
            return self.values[index]

    def sample(model, graph, key, *settings):
        calls.append((key, settings))
        if len(calls) == fail_on_batch:
            raise RuntimeError('synthetic batch failure')
        return Trajectory(key)

    generator.sample = sample
    return generator, calls


def test_avg_count_rounding_preserves_original_key_and_batch_sequence():
    generator, calls = mock_avg_generator()
    result = generator.generate('CCO', count=5, seed=7)
    assert [key for key, _ in calls] == [9, 10]
    assert all(settings == (2, 'uniform', 2., 'euler', 1e-5, 1e-4) for _, settings in calls)
    assert result['native_drawn_count'] == 8
    assert result['requested'] == result['output_count'] == 5
    assert result['coordinates'][:, 0, 0].tolist() == [9.]*4+[10.]
    assert result['raw_present_mask'].all() and result['present_mask'].all()


def test_avg_later_batch_failure_keeps_completed_raw_without_regenerating():
    generator, calls = mock_avg_generator(fail_on_batch=2)
    result = generator.generate('CCO', count=8, seed=7)
    assert len(calls) == 2
    assert result['raw_present_mask'].sum() == 4
    assert not result['present_mask'].any()
    assert 'synthetic batch failure' in result['failure']
    assert result['missing_output_slots'] == 8


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def mock_et_generator(monkeypatch, *, fail_after_first=False, reordered=False):
    generator = etflow.ETFlowGenerator.__new__(etflow.ETFlowGenerator)
    generator.metadata, generator.device = {}, 'cpu'
    generator.batch_size, generator.n_timesteps = 4, 50
    generator.featurizer = SimpleNamespace(get_data_from_smiles=lambda smiles: None)
    monkeypatch.setattr(etflow, 'validate_etflow_graph', lambda *args: None)

    def parse(smiles):
        molecule = input_molecule(smiles)
        return Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms())))) if reordered else molecule

    generator.parse_native = parse
    calls = []

    class Model:
        def switch_parity_of_pos(self, positions):
            return Tensor(-positions.values)

        def predict(self, smiles, **kwargs):
            calls.append(kwargs)
            n_atoms = input_molecule(smiles[0]).GetNumAtoms()
            output = []
            for offset in range(0, kwargs['num_samples'], kwargs['max_batch_size']):
                if offset and fail_after_first:
                    raise RuntimeError('synthetic batch failure')
                n = min(kwargs['max_batch_size'], kwargs['num_samples']-offset)
                raw = np.arange(n*n_atoms*3, dtype=float).reshape(n, n_atoms, 3)+offset
                output.append(self.switch_parity_of_pos(Tensor(raw)).numpy())
            return {smiles[0]: np.concatenate(output)}

    generator.model = Model()
    return generator, calls


def test_et_passes_native_sampling_settings_and_preserves_graph_mapping(monkeypatch):
    generator, calls = mock_et_generator(monkeypatch, reordered=True)
    original = generator.model.switch_parity_of_pos
    result = generator.generate('CCO', count=6, seed=17)
    assert calls == [dict(max_batch_size=4, num_samples=6, n_timesteps=50,
        seed=17, device='cpu', sampler_type='ode', as_mol=False)]
    assert generator.model.switch_parity_of_pos == original
    assert result['present_mask'].all() and result['raw_present_mask'].all()
    assert result['input_to_native'] == list(reversed(range(9)))
    np.testing.assert_array_equal(result['coordinates'], -result['raw_coordinates'])
    assert result['coordinates'][0, 0, 0] == -24.


def test_et_partial_native_batches_survive_failure_without_retry(monkeypatch):
    generator, calls = mock_et_generator(monkeypatch, fail_after_first=True)
    result = generator.generate('CCO', count=6)
    assert len(calls) == 1
    assert result['present_mask'].tolist() == [True]*4+[False]*2
    assert np.isnan(result['coordinates'][4:]).all()
    assert 'synthetic batch failure' in result['failure']


def test_et_native_changed_chemical_state_fails_before_sampling(monkeypatch):
    generator, calls = mock_et_generator(monkeypatch)
    generator.parse_native = lambda smiles: input_molecule('CC[O-]')
    result = generator.generate('CCO', count=3)
    assert not calls and not result['present_mask'].any()
    assert result['failure'].startswith('UnsupportedGraph:')


def fake_conforge(monkeypatch, *, status=0):
    settings_marker = object()
    instances = []

    class Settings:
        def assign(self, value):
            assert value is settings_marker

        def clearMaxNumOutputConformersRanges(self):
            self.ranges_cleared = True

    class Conformer:
        energy = 4.

        def __iter__(self):
            return iter(SimpleNamespace(toArray=lambda i=i: np.array([i, i+1, i+2.])) for i in range(9))

    class Generator:
        def __init__(self):
            self.settings, self.numConformers = Settings(), 2
            instances.append(self)

        def setLogMessageCallback(self, callback):
            self.log = callback

        def setTimeoutCallback(self, callback):
            self.timeout = callback

        def generate(self, molecule):
            self.log('synthetic native call')
            return status

        def getConformer(self, index):
            return Conformer()

    confgen = SimpleNamespace(ConformerGenerator=Generator,
        ConformerGeneratorSettings=SimpleNamespace(MEDIUM_SET_DIVERSE=settings_marker),
        ConformerSamplingMode=SimpleNamespace(AUTO=3),
        ReturnCode=SimpleNamespace(SUCCESS=0, TOO_MUCH_SYMMETRY=1, TIMEOUT=2))
    monkeypatch.setattr(conforge, 'cdpl_modules', lambda source=None: (None, confgen))
    monkeypatch.setattr(conforge, 'prepare_mapped_graph', lambda mol, source=None:
        (object(), np.arange(mol.GetNumAtoms())[::-1], 'mapped-input'))
    return instances


def test_conforge_preserves_native_preset_mapping_and_partial_output(monkeypatch):
    instances = fake_conforge(monkeypatch)
    result = conforge.generate('CCO', count=5, timeout_ms=13)
    settings = instances[0].settings
    assert settings.maxNumOutputConformers == 5 and settings.timeout == 13
    assert settings.genCoordsFromScratch and not settings.includeInputCoords
    assert result['coordinates'][0, 0].tolist() == [8., 9., 10.]
    assert result['present_mask'].tolist() == [True, True, False, False, False]
    assert result['missing_output_slots'] == 3
    assert not result['seed_controlled'] and result['public_seed_parameter'] is None


def test_conforge_native_timeout_remains_failure_with_fixed_denominator(monkeypatch):
    instances = fake_conforge(monkeypatch, status=2)
    result = conforge.generate('CCO', count=5)
    assert len(instances) == 1
    assert result['failure'] == 'native_TIMEOUT'
    assert result['missing_output_slots'] == 5 and not result['present_mask'].any()


def test_conforge_nonzero_seed_not_silently_ignored():
    with pytest.raises(ValueError, match='no seed setter'):
        conforge.generate('CCO', seed=1)


def test_missing_optional_dependency_reports_installation_requirement(monkeypatch):
    original_import = builtins.__import__

    def missing_cdpl(name, *args, **kwargs):
        if name.startswith('CDPL'):
            raise ModuleNotFoundError('synthetic absent CDPL')
        return original_import(name, *args, **kwargs)

    conforge.cdpl_modules.cache_clear()
    monkeypatch.setattr(builtins, '__import__', missing_cdpl)
    with pytest.raises(ImportError, match='CONFORGE requires CDPKit 1.3.0'):
        conforge.cdpl_modules()


@pytest.mark.parametrize('module', [avgflow, etflow])
def test_neural_paths_and_device_are_explicit(module, tmp_path):
    cls = module.AvgFlowGenerator if module is avgflow else module.ETFlowGenerator
    with pytest.raises(TypeError):
        cls()
    kwargs = dict(source=tmp_path, checkpoint=tmp_path/'missing', device='cpu')
    if module is avgflow:
        kwargs['config'] = tmp_path/'missing.yaml'
    with pytest.raises(FileNotFoundError, match='local checkout'):
        cls(**kwargs)


def test_neural_public_calls_accept_no_target_arguments():
    for cls in (avgflow.AvgFlowGenerator, etflow.ETFlowGenerator):
        generator = cls.__new__(cls)
        with pytest.raises(TypeError):
            generator.generate('CCO', target_coordinates=np.zeros((9, 3)))
