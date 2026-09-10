"""Synthetic CPU-only generation and fixed-ID relaxation contracts."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conformer_fidelity.generators import etkdg
from conformer_fidelity.generators._common import (
    UnsupportedGraph, coordinate_identity, fixed_slots, input_molecule,
    input_to_native_mapping,
)


def coordinates(smiles='CCO', count=3):
    molecule = input_molecule(smiles)
    assert AllChem.EmbedMolecule(molecule, randomSeed=131) == 0
    return np.repeat(molecule.GetConformer().GetPositions()[None], count, axis=0)


def test_etkdg_parameters_and_float32_materialization():
    result = etkdg.generate('CCCO', count=3, seed=29)
    expected = input_molecule('CCCO')
    params = AllChem.ETKDGv3()
    params.randomSeed, params.numThreads = 29, 1
    params.clearConfs, params.enforceChirality = True, True
    ids = AllChem.EmbedMultipleConfs(expected, numConfs=3, params=params)
    expected_xyz = np.asarray([expected.GetConformer(i).GetPositions() for i in ids], dtype=np.float32)
    np.testing.assert_array_equal(result['coordinates'], expected_xyz)
    assert result['present_mask'].tolist() == [True]*3
    assert result['requested'] == result['output_count'] == 3
    assert result['molecule'].GetNumConformers() == 0
    assert not result['target_access'] and not result['external_optimization']


def test_missing_embedding_slots_are_not_retried(monkeypatch):
    original = AllChem.EmbedMultipleConfs
    calls = []

    def partial(molecule, *, numConfs, params):
        calls.append(numConfs)
        return original(molecule, numConfs=1, params=params)

    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', partial)
    result = etkdg.generate('CCO', count=4)
    assert calls == [4]
    assert result['present_mask'].tolist() == [True, False, False, False]
    assert np.isnan(result['coordinates'][1:]).all()
    assert result['missing_output_slots'] == 3


def test_returned_nonfinite_slot_is_present_not_padding():
    molecule = input_molecule('CCO')
    result = fixed_slots(dict(molecule=molecule,
        coordinates=np.full((1, molecule.GetNumAtoms(), 3), np.nan)), 3)
    assert result['present_mask'].tolist() == [True, False, False]
    assert result['output_count'] == 1


@pytest.mark.parametrize('bad', ['invalid', 'CC.CC', 'CC |(0,0,0;1,1,1)|'])
def test_coordinate_or_disconnected_inputs_rejected(bad):
    with pytest.raises(ValueError):
        etkdg.generate(bad, count=1)


@pytest.mark.parametrize('kwargs', [{'count': 0}, {'seed': -1}, {'seed': 2**31}])
def test_request_validation(kwargs):
    with pytest.raises(ValueError):
        etkdg.generate('CC', **kwargs)


def test_mapping_allows_only_coordinate_free_exact_chemistry():
    original = input_molecule('[13CH3:7][C@H:8](O)CC')
    order = list(reversed(range(original.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(original, order)
    mapping = input_to_native_mapping(original, reordered)
    assert mapping.tolist() == order
    changed = input_molecule('[13CH3:7][C@@H:8](O)CC')
    with pytest.raises(UnsupportedGraph):
        input_to_native_mapping(original, changed)
    charged = input_molecule('[13CH3:7][C@H:8]([O-])CC')
    with pytest.raises(UnsupportedGraph):
        input_to_native_mapping(original, charged)


def test_stereo_diagnostic_reads_geometry_not_inherited_ez():
    molecule = input_molecule('F/C=C/F')
    cis = np.array([[-1., 1., 0.], [0., 0., 0.], [1.3, 0., 0.],
                    [2.3, 1., 0.], [-.3, -1., 0.], [1.6, -1., 0.]])
    assert not coordinate_identity(molecule, cis[None])[0]['specified_stereo_retained']


def test_relax_preserves_missing_nonfinite_and_distorted_slots():
    xyz = coordinates(count=4)
    xyz[0, 0, 0] += 2.0  # No initial geometry eligibility filter.
    xyz[2] = np.nan
    original = xyz.copy()
    result = etkdg.relax('CCO', xyz, present_mask=[True, False, True, True])
    assert result['coordinates'].shape == xyz.shape
    assert result['present_mask'].tolist() == [True, False, True, True]
    assert result['optimizer_attempted'] == [True, False, False, True]
    assert result['optimizer_failure'][1:3] == ['native_missing_slot', 'native_nonfinite_coordinates']
    assert np.isnan(result['coordinates'][1:3]).all()
    assert np.isfinite(result['coordinates'][[0, 3]]).all()
    np.testing.assert_array_equal(xyz, original)


def test_status1_retains_actual_coordinates_and_explicit_iteration_budget(monkeypatch):
    class ForceField:
        def Initialize(self):
            pass

        def Minimize(self, maxIts):
            assert maxIts == 17
            return 1

        def CalcEnergy(self):
            return 5.

    monkeypatch.setattr(AllChem, 'MMFFGetMoleculeForceField', lambda *a, **k: ForceField())
    xyz = coordinates(count=2)
    result = etkdg.relax('CCO', xyz, max_iterations=17)
    np.testing.assert_array_equal(result['coordinates'], xyz)
    assert result['optimizer_status'] == [1, 1]
    assert result['optimizer_failure'] == [None, None]
    assert result['forcefield'] == 'MMFF94s'


def test_missing_parameters_never_fall_back_to_native(monkeypatch):
    monkeypatch.setattr(AllChem, 'MMFFGetMoleculeProperties', lambda *a, **k: None)
    result = etkdg.relax('CCO', coordinates(count=2))
    assert np.isnan(result['coordinates']).all()
    assert result['present_mask'].all()
    assert result['optimizer_attempted'] == [False, False]
    assert all(f.startswith('mmff_setup_failed') for f in result['optimizer_failure'])


def test_optimizer_exception_keeps_failed_slot_and_other_success(monkeypatch):
    original = AllChem.MMFFGetMoleculeForceField
    calls = []

    def forcefield(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('synthetic optimizer failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(AllChem, 'MMFFGetMoleculeForceField', forcefield)
    result = etkdg.relax('CCO', coordinates(count=2))
    assert np.isnan(result['coordinates'][0]).all()
    assert np.isfinite(result['coordinates'][1]).all()
    assert 'synthetic optimizer failure' in result['optimizer_failure'][0]


def test_generation_and_relaxation_do_not_accept_target_inputs():
    with pytest.raises(TypeError):
        etkdg.generate('CCO', target_coordinates=coordinates()[0])
    with pytest.raises(TypeError):
        etkdg.relax('CCO', coordinates(), target_coordinates=coordinates()[0])
