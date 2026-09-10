import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from conformer_fidelity.chemistry import graph_context, input_molecule
from conformer_fidelity.evaluation import score_pool
from conformer_fidelity.selection import METHODS, pool_geometry, select_conformers, select_indices


def line_distances(points):
    x = np.asarray(points, dtype=float)
    return np.abs(x[:, None] - x)


def test_facility_and_energy_facility_preserve_fixed_rank_rule():
    distances = line_distances([0., 1., 2., 10.])
    energies = np.array([-10., 0., 1., 2.])
    assert select_indices(distances, energies, 'facility', 2).tolist() == [1, 3]
    # First-step mixed ranks are [2/3, 3/4, 7/12, 0]: the medoid wins
    # before the lowest-energy endpoint, unlike pure energy ordering.
    assert select_indices(distances, energies, 'energy_facility', 2).tolist() == [1, 0]
    assert select_indices(distances, energies, 'energy', 2).tolist() == [0, 1]
    assert select_indices(distances, energies, 'kcenter', 2).tolist() == [0, 3]


@pytest.mark.parametrize('method', METHODS)
def test_ties_empty_zero_and_short_pools(method):
    d = np.zeros((3, 3))
    e = np.zeros(3)
    result = select_indices(d, e, method)
    expected = [0] if method.startswith('energy_rmsd') else [0, 1, 2]
    assert result.tolist() == expected
    assert select_indices(d, e, method, 0).size == 0
    assert select_indices(np.empty((0, 0)), [], method).size == 0
    assert result.dtype == np.int64 and len(np.unique(result)) == len(result)


def test_radius_is_strict_and_short_outputs_are_not_refilled():
    distances = line_distances([0., .5, 1., 1.5])
    energies = np.arange(4.)
    assert select_indices(distances, energies, 'energy_rmsd05').tolist() == [0, 2]
    assert select_indices(distances, energies, 'energy_rmsd10').tolist() == [0, 3]


@pytest.mark.parametrize('method', METHODS)
def test_fixed_slot_masks_ignore_placeholder_features_and_preserve_indices(method):
    d = np.full((5, 5), np.nan)
    e = np.full(5, np.nan)
    valid = [1, 4]
    d[np.ix_(valid, valid)] = [[0., 2.], [2., 0.]]
    e[valid] = [3., 0.]
    present = np.array([False, True, True, False, True])
    eligible = np.array([True, True, False, False, True])
    original_d, original_e = d.copy(), e.copy()
    result = select_indices(d, e, method, present_mask=present, eligible_mask=eligible)
    assert set(result.tolist()) == {1, 4}
    np.testing.assert_array_equal(d, original_d)
    np.testing.assert_array_equal(e, original_e)


def test_invalid_eligible_features_and_target_arguments_are_rejected():
    d = line_distances([0., 1.])
    with pytest.raises(ValueError, match='finite'):
        select_indices(d, [0., np.nan])
    with pytest.raises(ValueError, match='symmetric'):
        select_indices([[0., 2.], [1., 0.]], [0., 1.])
    with pytest.raises(ValueError, match='boolean'):
        select_indices(d, [0., 1.], present_mask=[1, 1])
    with pytest.raises(ValueError, match='maximum'):
        select_indices(d, [0., 1.], maximum=21)
    for function in (select_indices, pool_geometry, select_conformers):
        assert not any('target' in name for name in inspect.signature(function).parameters)
    with pytest.raises(TypeError, match='target'):
        select_indices(d, [0., 1.], target_distances=[0., 1.])


def test_selection_import_does_not_require_training_or_torch():
    code = """
import sys
sys.modules['torch'] = None
sys.modules['sklearn'] = None
from conformer_fidelity.selection import select_indices
assert select_indices([[0., 1.], [1., 0.]], [0., 1.]).tolist() == [0, 1]
"""
    subprocess.run([sys.executable, '-c', code], env=os.environ.copy(), check=True,
                   capture_output=True, text=True)


def test_synthetic_conformers_match_feature_selection_and_keep_missing_slots():
    smiles = 'CCCC'
    molecule = input_molecule(smiles)
    assert AllChem.EmbedMolecule(molecule, randomSeed=34) == 0
    AllChem.MMFFOptimizeMolecule(molecule, mmffVariant='MMFF94s', maxIters=200)
    context = graph_context({'smiles': smiles})
    rotor = context['rotors'][0]
    candidates = []
    for angle in (-180., -60., 60.):
        work = Chem.Mol(molecule)
        rdMolTransforms.SetDihedralDeg(work.GetConformer(), *rotor.dihedral, angle)
        candidates.append(work.GetConformer().GetPositions())
    candidates = np.stack([candidates[0], candidates[0], candidates[1], candidates[2]])
    present = np.array([False, True, True, True])
    before = candidates.copy()
    features = pool_geometry(smiles, candidates, present)
    assert np.isnan(features['energies'][0])
    assert np.isnan(features['distance'][0]).all()
    assert features['eligible_mask'].tolist() == present.tolist()
    for method in METHODS:
        selected = select_conformers(smiles, candidates, method, 2, present)
        expected = select_indices(features['distance'], features['energies'], method, 2,
                                  present, features['eligible_mask'])
        np.testing.assert_array_equal(selected['indices'], expected)
        assert set(expected.tolist()).issubset({1, 2, 3})
        assert selected['target_access'] is selected['coordinates_changed'] is selected['refill'] is False
        assert selected['maximum'] == 2 and selected['selected_count'] <= 2
    np.testing.assert_array_equal(candidates, before)
    chosen = select_conformers(smiles, candidates, 'energy_rmsd10', 2, present)['indices']
    # Selection and scoring remain separate; absent output slots stay absent.
    slots = np.full((20, molecule.GetNumAtoms(), 3), np.nan)
    slots[:len(chosen)] = candidates[chosen]
    mask = np.arange(20) < len(chosen)
    scored = score_pool(smiles, slots, candidates[1], mask)
    assert scored['output_count'] == len(chosen)
    assert scored['output_fraction'] == len(chosen)/20
