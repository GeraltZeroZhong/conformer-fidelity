"""Graph inputs and coordinate-dependent chemical eligibility."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SanitizeFlags

from .rotors import extract_torsions, find_bridge_rotors
from .stereo import stereochemistry_retention, stereochemistry_signature
from .symmetry import heavy_atom_automorphisms

TOPOLOGY_TOLERANCE = 0.40
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}


def input_molecule(smiles):
    """Build the supported all-H graph from SMILES only, without coordinates.

    Atom order is exactly ``Chem.AddHs(Chem.MolFromSmiles(smiles))``. The
    supported graph requires one fragment, supported elements, no radicals or
    atropisomeric bonds, and MMFF94s parameters. No embedding is performed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid_smiles')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('disconnected_graph')
    mol = Chem.AddHs(mol)
    if any(a.GetAtomicNum() not in COVALENT_RADII for a in mol.GetAtoms()):
        raise ValueError('unsupported_element')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('radical_graph')
    if any('ATROP' in str(b.GetStereo()).upper() for b in mol.GetBonds()):
        raise ValueError('unsupported_atropisomerism')
    if AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s') is None:
        raise ValueError('mmff_parameters_unavailable')
    if mol.GetNumConformers():
        raise ValueError('input unexpectedly contains coordinates')
    return mol


def _sanitized_single_fragment(molecule):
    try:
        result = Chem.Mol(molecule)
        Chem.SanitizeMol(result, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        Chem.Kekulize(result, clearAromaticFlags=True)
    except Exception:
        return None
    return result if len(Chem.GetMolFrags(result)) == 1 else None


def topology_is_valid(molecule, positions, *, tolerance=TOPOLOGY_TOLERANCE):
    """Check distances for covalent bonds declared in the input graph.

    Coordinates have shape [conformers, all-H atoms, 3]. Each bond is compared
    with the sum of its atoms' covalent radii, including bonded hydrogens.
    """
    if hasattr(positions, 'detach'):
        positions = positions.detach().cpu().numpy()
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (molecule.GetNumAtoms(), 3):
        return False
    if not np.isfinite(coordinates).all():
        return False
    numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64)
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        expected = COVALENT_RADII.get(int(numbers[left]), 1.5) + COVALENT_RADII.get(
            int(numbers[right]), 1.5
        )
        observed = np.linalg.norm(coordinates[:, left] - coordinates[:, right], axis=-1)
        if not np.all(np.abs(observed - expected) <= expected * tolerance):
            return False
    return True


def graph_context(row):
    """Build atom identity, heavy permutations, rotors and specified stereo.

    Only ``row['smiles']`` is required. If ``heavy_atoms`` is present, its count
    is checked. No seed, target coordinates or other row metadata are consumed.
    """
    mol = input_molecule(row['smiles'])
    if _sanitized_single_fragment(mol) is None:
        raise ValueError('invalid input graph')
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if 'heavy_atoms' in row and len(heavy) != row['heavy_atoms']:
        raise ValueError('heavy atom count mismatch')
    if not heavy:
        raise ValueError('at least one heavy atom required')
    lookup = {j: i for i, j in enumerate(heavy)}
    mappings = heavy_atom_automorphisms(mol, max_automorphisms=16385)
    if len(mappings) > 16384:
        raise ValueError('automorphism capacity exceeded')
    permutations = tuple(tuple(lookup[p[i]] for i in heavy) for p in mappings)
    return dict(molecule=mol, heavy=heavy, permutations=permutations,
                rotors=find_bridge_rotors(mol), stereo=stereochemistry_signature(mol))


def original_eligibility(context, xyz):
    """Check float32 finiteness, bond distances, specified stereo and torsions.

    Returns None on success or the first failure reason. Independent stereo
    diagnosis is separate and does not stop at a bonded-distance failure.
    """
    import torch

    mol = context['molecule']
    positions = torch.as_tensor(np.asarray(xyz), dtype=torch.float32)
    if positions.shape != (mol.GetNumAtoms(), 3):
        raise ValueError('coordinate shape or atom order differs from input graph')
    if not torch.isfinite(positions).all():
        return 'nonfinite_positions'
    if not topology_is_valid(mol, positions[None], tolerance=TOPOLOGY_TOLERANCE):
        return 'topology_change'
    observed = stereochemistry_signature(mol, positions=positions, from_geometry=True)
    if not stereochemistry_retention(context['stereo'], observed).all_retained:
        return 'stereo_change'
    if not torch.isfinite(extract_torsions(positions, context['rotors'])).all():
        return 'nonfinite_torsions'
    return None
