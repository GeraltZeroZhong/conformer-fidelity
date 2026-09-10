import json

import numpy as np
import pytest
import torch
from rdkit.Chem import AllChem

from conformer_fidelity.chemistry import graph_context, input_molecule
from conformer_fidelity.evaluation import FIELDS, pairwise_ensemble_rmsd, score_pool, symmetry_aware_rmsd


def example():
    mol = input_molecule('CCO')
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')
    return mol.GetConformer().GetPositions()


def test_public_all_h_and_explicit_heavy_targets_match():
    xyz = example()
    before = xyz.copy()
    heavy = graph_context({'smiles': 'CCO'})['heavy']
    all_h = score_pool('CCO', xyz[None], xyz)
    only_heavy = score_pool('CCO', xyz[None], xyz[heavy])
    assert {f: all_h[f] for f in FIELDS} == {f: only_heavy[f] for f in FIELDS}
    assert all_h['finite_hit_1'] == all_h['original_hit_1'] == all_h['qhit_1'] == 1
    assert all_h['geometry_fraction'] == .05
    assert all_h['qualified_output_fraction'] == 1
    assert len(all_h['geometry'][0]['tests']) == 6
    json.dumps(all_h, allow_nan=False)
    np.testing.assert_array_equal(xyz, before)


def test_absent_finite_and_present_nonfinite_slots_are_distinct():
    xyz = example()
    candidates = np.stack([xyz, xyz, np.full_like(xyz, np.nan)])
    before = candidates.copy()
    result = score_pool('CCO', candidates, xyz, [False, True, True])
    assert result['present_mask'] == [False, True, True]
    assert result['finite_mask'] == [False, True, False]
    assert result['output_count'] == 2 and result['output_fraction'] == .1
    assert result['finite_count'] == result['qualified_count'] == 1
    assert result['qualified_output_fraction'] == .5
    assert result['nonfinite_count'] == 1
    assert result['target_distance'][0] == result['target_distance'][2] == 100
    np.testing.assert_array_equal(candidates, before)


def test_empty_pool_and_invalid_targets_are_not_imputed():
    xyz = example()
    value = score_pool('CCO', np.empty((0, len(xyz), 3)), xyz)
    assert value['output_count'] == 0 and value['missing'] == 20
    for prefix in ('finite', 'original', 'qualified'):
        assert value[prefix+'_best_rmsd'] == 100
    bad = xyz.copy()
    bad[-1] = np.nan
    with pytest.raises(ValueError, match='target must be finite'):
        score_pool('CCO', xyz[None], bad)
    with pytest.raises(ValueError, match='target must be finite'):
        score_pool('CCO', xyz[None], xyz[:2])
    with pytest.raises(ValueError, match='denominator'):
        score_pool('CCO', np.repeat(xyz[None], 2, axis=0), xyz, denominator=1)
    with pytest.raises(ValueError, match='boolean'):
        score_pool('CCO', xyz[None], xyz, [1])


def test_three_nested_levels_use_all_finite_distances_and_float32(monkeypatch):
    import conformer_fidelity.evaluation.pool as module

    xyz = np.array([[[0., 0., 0.], [1.000000001, 0., 0.]],
                    [[0., 0., 0.], [2., 0., 0.]],
                    [[0., 0., 0.], [3., 0., 0.]],
                    [[np.nan, 0., 0.], [4., 0., 0.]]])
    observation = dict(context=dict(heavy=[0, 1], permutations=((0, 1),)),
        coordinates=xyz, eligible=np.array([False, True, True, False]),
        failures=['stereo_change', None, None, 'nonfinite_positions'],
        geometry=[dict(passed=False, seconds=0., tests={}),
                  dict(passed=False, seconds=0., tests={'bond': False}),
                  dict(passed=True, seconds=0., tests={'bond': True}),
                  dict(passed=False, seconds=0., tests={})])
    seen = []

    def distance(probe, target, permutations):
        seen.append(probe.clone())
        assert probe.dtype == target.dtype == torch.float64
        assert probe[0, 1, 0].item() == 1.0
        assert len(probe) == 3
        return torch.tensor([[.25], [1.5], [2.5]], dtype=torch.float64)

    monkeypatch.setattr(module, 'pairwise_ensemble_rmsd', distance)
    distances = module.target_distances(observation, xyz[0])
    result = module.score_fixed_selection(observation, [0, 1, 2, 3], xyz[0], target_distance=distances)
    assert len(seen) == 1
    np.testing.assert_array_equal(distances, [.25, 1.5, 2.5, 100.])
    assert (result['finite_hit_1'], result['original_hit_1'], result['qhit_1']) == (1, 0, 0)
    assert (result['finite_hit_2'], result['original_hit_2'], result['qhit_2']) == (1, 1, 0)
    assert result['original_hit_loss_1'] == result['geometry_hit_loss_2'] == 1
    assert result['qualified_count'] == 1 and result['missing'] == 18


def test_rigid_alignment_uses_no_reflection_and_honors_symmetry():
    probe = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], dtype=torch.float64)
    mirror = probe.clone()
    mirror[:, 0] *= -1
    assert symmetry_aware_rmsd(probe, mirror).item() > .1
    order = (0, 2, 1, 3)
    permuted = probe[list(order)]
    value = symmetry_aware_rmsd(probe, permuted, permutations=[tuple(range(4)), order])
    assert value.item() < 1e-12
    matrix = pairwise_ensemble_rmsd(probe[None], permuted[None], permutations=[order], pair_chunk_size=1)
    assert matrix.shape == (1, 1) and matrix.item() < 1e-12
