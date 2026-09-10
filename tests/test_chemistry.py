import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conformer_fidelity.chemistry import (
    find_bridge_rotors,
    graph_context,
    independent_stereo,
    input_molecule,
    original_eligibility,
    stereochemistry_retention,
    stereochemistry_signature,
)


def embedded(smiles):
    mol = input_molecule(smiles)
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=200)
    return mol, mol.GetConformer().GetPositions()


def test_graph_only_inputs_and_optional_count():
    first = graph_context({'smiles': 'CCCC'})
    second = graph_context({'smiles': 'CCCC', 'ignored_target': object()})
    assert first['molecule'].GetNumConformers() == 0
    assert first['heavy'] == [0, 1, 2, 3]
    assert first['permutations'] == second['permutations']
    assert len(first['rotors']) == 1
    assert first['rotors'][0].dihedral == second['rotors'][0].dihedral
    with pytest.raises(ValueError, match='count'):
        graph_context({'smiles': 'CCCC', 'heavy_atoms': 3})
    with pytest.raises(ValueError, match='disconnected'):
        input_molecule('CC.O')
    with pytest.raises(ValueError, match='radical'):
        input_molecule('[CH3]')


def test_import_and_graph_stereo_do_not_require_torch():
    code = """
import sys
sys.modules['torch'] = None
from conformer_fidelity.chemistry import graph_context, independent_stereo
assert graph_context({'smiles': 'CC'})['heavy'] == [0, 1]
assert sys.modules['torch'] is None
"""
    subprocess.run([sys.executable, '-c', code], env=os.environ.copy(), check=True,
                   capture_output=True, text=True)


def test_mirror_reassigned_without_mutating_reference():
    mol, xyz = embedded('C[C@H](O)C(=O)O')
    reference = stereochemistry_signature(mol)
    observed = stereochemistry_signature(mol, positions=xyz, from_geometry=True)
    assert stereochemistry_retention(reference, observed).all_retained
    mirror = xyz.copy()
    mirror[:, 0] *= -1
    result = independent_stereo(mol, mirror)
    assert result['specified_stereo_retained'] is False
    assert result['rs_retained'] is False
    assert result['ez_retained'] is None
    assert result['rs_mismatches']
    assert stereochemistry_signature(mol) == reference
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), xyz)


def test_independent_stereo_does_not_short_circuit_on_bond_failure():
    mol, xyz = embedded('C[C@H](O)C(=O)O')
    xyz[:, 0] *= -1
    xyz *= 2
    context = graph_context({'smiles': 'C[C@H](O)C(=O)O'})
    assert original_eligibility(context, xyz) == 'topology_change'
    assert independent_stereo(mol, xyz)['specified_stereo_retained'] is False


def test_ez_geometry_sentinel_and_vacuous_reference():
    mol = input_molecule('F/C=C/F')
    cis = np.array([[-1, 1, 0], [0, 0, 0], [1.3, 0, 0], [2.3, 1, 0],
                    [-.3, -1, 0], [1.6, -1, 0]], dtype=float)
    assert stereochemistry_signature(mol, positions=cis, from_geometry=True).double_bonds == ((1, 2, 'Z'),)
    result = independent_stereo(mol, cis)
    assert result['ez_retained'] is False
    assert result['observed_ez'] == [[1, 2, 'Z']]
    ordinary, xyz = embedded('CCO')
    assert independent_stereo(ordinary, xyz)['specified_stereo_retained'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        independent_stereo(mol, np.full_like(cis, np.nan))


def test_strict_rotors_and_heavy_symmetry():
    assert not find_bridge_rotors(input_molecule('CC(=O)NC'))
    benzene = graph_context({'smiles': 'c1ccccc1'})
    assert len(benzene['permutations']) == 12
    assert benzene['permutations'][0] == tuple(range(6))
    assert Chem.MolToSmiles(benzene['molecule'])
