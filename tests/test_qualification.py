import numpy as np
import pytest
from rdkit.Chem import AllChem

from conformer_fidelity.chemistry import input_molecule
from conformer_fidelity.evaluation import configuration, metrics, observe


def example():
    mol = input_molecule('CCO')
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')
    return mol, mol.GetConformer().GetPositions()


def test_six_original_geometry_checks_on_small_molecule():
    mol, xyz = example()
    config = configuration()
    assert len(config['modules']) == 4
    assert config['additional_energy_sampling'] is False
    assert config['target_input'] is False
    native = observe(mol, xyz[None], config=config)[0]
    assert native['passed'] is True
    assert len(native['tests']) == 6
    distorted = xyz.copy()
    distorted[0, 0] += 20
    assert observe(mol, distorted[None], config=config)[0]['passed'] is False


def test_geometry_never_overrides_original_failure():
    mol, xyz = example()
    row = observe(mol, xyz[None], config=configuration(), eligible=[False])[0]
    assert not row['passed']
    assert row['tests'] == {}
    assert row['error'] == 'original_eligibility_failure_or_nonfinite'


def test_missing_binary_result_is_not_truthy_nan(monkeypatch):
    import conformer_fidelity.evaluation.qualification as module

    mol, xyz = example()
    config = dict(modules=[dict(name='Geometry', function='distance_geometry',
                               parameters={}, chosen_binary_test_output=['valid'])])
    monkeypatch.setattr(module, 'check_geometry', lambda *a, **k: {'results': {'valid': np.nan}})
    row = observe(mol, xyz[None], config=config)[0]
    assert row['tests'] == {'Geometry/valid': False}
    assert row['passed'] is False


def test_same_candidate_must_pass_and_hit_with_fixed_denominator():
    geometry = [dict(passed=False, seconds=0.), dict(passed=True, seconds=0.)]
    value = metrics([.2, 4.], [0, 1], geometry, [True, True], [0, 1])
    assert value['raw_hit_1'] == 1
    assert value['qhit_1'] == value['qhit_2'] == 0
    assert value['qualified_best_rmsd'] == 4
    assert value['geometry_fraction'] == .05
    assert value['eligible_fraction'] == .1
    assert value['missing'] == 18
    with pytest.raises(ValueError, match='selection'):
        metrics([.2, 4.], [0, 0], geometry, [True, True], [0, 1])


def test_empty_selection_keeps_penalty():
    result = metrics([], [], [], [], [])
    assert result['qhit_1'] == result['qhit_2'] == 0
    assert result['best_rmsd'] == result['qualified_best_rmsd'] == 100
    assert result['missing'] == 20
