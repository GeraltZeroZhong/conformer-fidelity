"""Shared coordinate-free input, exact atom mapping and fixed-slot results."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

def input_molecule(smiles):
    if not isinstance(smiles, str) or not smiles or '|' in smiles or any(c.isspace() for c in smiles):
        raise ValueError('plain coordinate-free 2D SMILES required')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('invalid/disconnected input graph')
    mol = Chem.AddHs(mol)
    if mol.GetNumConformers():
        raise ValueError('input contains coordinates')
    return mol


def graph_record(mol):
    """Portable indexed graph; no RDKit binary or coordinates required to read it."""
    return dict(atoms=[dict(atomic_num=a.GetAtomicNum(), formal_charge=a.GetFormalCharge(),
        isotope=a.GetIsotope(), atom_map_num=a.GetAtomMapNum(), chiral_tag=str(a.GetChiralTag()),
        num_radical_electrons=a.GetNumRadicalElectrons(),
        total_h_count=a.GetTotalNumHs(includeNeighbors=True), is_aromatic=a.GetIsAromatic())
        for a in mol.GetAtoms()], bonds=[dict(begin=b.GetBeginAtomIdx(), end=b.GetEndAtomIdx(),
        bond_type=str(b.GetBondType()), is_aromatic=b.GetIsAromatic(), stereo=str(b.GetStereo()),
        stereo_atoms=list(b.GetStereoAtoms())) for b in mol.GetBonds()],
        heavy_indices=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])


class UnsupportedGraph(ValueError):
    """The backend graph differs from the specified input chemistry."""


def _chemical_smiles(mol):
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copied, canonical=True, isomericSmiles=True, allHsExplicit=True)


def _mapping_preserves_atoms_and_bonds(first, second, mapping):
    if len(mapping) != len(first['atoms']) or sorted(mapping) != list(range(len(second['atoms']))):
        return False
    for i, j in enumerate(mapping):
        a, b = first['atoms'][i], second['atoms'][j]
        if any(a[k] != b[k] for k in a if k != 'chiral_tag'):
            return False
    def bonds(record, permutation):
        return sorted((min(permutation[b['begin']], permutation[b['end']]),
            max(permutation[b['begin']], permutation[b['end']]), b['bond_type'], b['is_aromatic'])
            for b in record['bonds'])
    return bonds(first, mapping) == bonds(second, list(range(len(second['atoms']))))


def input_to_native_mapping(input_mol, native_mol):
    """Exact 2D chemistry first; deterministic coordinate-free isomorphism only.

    The normal official path is the identity mapping. Atom reordering may be
    mapped, but charge/H/isotope/chemical stereo changes are never repaired.
    """
    if native_mol.GetNumConformers():
        raise UnsupportedGraph('native graph unexpectedly carries coordinates before generation')
    if (input_mol.GetNumAtoms() != native_mol.GetNumAtoms()
            or input_mol.GetNumBonds() != native_mol.GetNumBonds()
            or _chemical_smiles(input_mol) != _chemical_smiles(native_mol)):
        raise UnsupportedGraph('native graph changed atom inventory, charge/H/isotope or specified stereo')
    first, second = graph_record(input_mol), graph_record(native_mol)
    identity = list(range(input_mol.GetNumAtoms()))
    if _mapping_preserves_atoms_and_bonds(first, second, identity):
        # Atom-indexed stereochemistry must also agree, not just molecular identity.
        tagged_a, tagged_b = Chem.Mol(input_mol), Chem.Mol(native_mol)
        for i in identity:
            tagged_a.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
            tagged_b.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
        if Chem.MolToSmiles(tagged_a, isomericSmiles=True) == Chem.MolToSmiles(tagged_b, isomericSmiles=True):
            return np.asarray(identity, dtype=np.int64)
    mapping = list(native_mol.GetSubstructMatch(input_mol, useChirality=True))
    if not _mapping_preserves_atoms_and_bonds(first, second, mapping):
        raise UnsupportedGraph('no exact atom-identity-preserving 2D mapping')
    # Verify indexed stereo under the chosen mapping; no coordinate-derived tie break.
    reordered = Chem.RenumberAtoms(native_mol, mapping)
    tagged_a, tagged_b = Chem.Mol(input_mol), Chem.Mol(reordered)
    for i in range(input_mol.GetNumAtoms()):
        tagged_a.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
        tagged_b.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
    if Chem.MolToSmiles(tagged_a, isomericSmiles=True) != Chem.MolToSmiles(tagged_b, isomericSmiles=True):
        raise UnsupportedGraph('native mapping changes indexed specified stereochemistry')
    return np.asarray(mapping, dtype=np.int64)


def validate_request(count, seed):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('count must be a positive integer')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')


def fixed_slots(result, count):
    """Pad only missing requests; returned nonfinite coordinates remain present."""
    molecule = result.get('molecule')
    n_atoms = molecule.GetNumAtoms() if molecule is not None else 0
    coordinates = np.asarray(result['coordinates'], dtype=np.float64)
    if coordinates.shape == (0, 0, 3) and n_atoms:
        coordinates = np.empty((0, n_atoms, 3))
    if coordinates.ndim != 3 or coordinates.shape[1:] != (n_atoms, 3):
        raise ValueError('coordinates must follow the input all-H atom order')
    observed = len(coordinates)
    if observed > count:
        raise ValueError('native generator returned more than requested')
    output = np.full((count, n_atoms, 3), np.nan, dtype=np.float64)
    output[:observed] = coordinates
    present = np.zeros(count, dtype=bool)
    present[:observed] = True
    result.update(coordinates=output, present_mask=present, requested=count,
        output_count=observed, missing_output_slots=count-observed, target_access=False,
        atom_order='RDKit AddHs(MolFromSmiles(input)); all atoms including explicit H')
    if molecule is not None:
        result['input_graph'] = graph_record(molecule)
    if 'raw_coordinates' in result:
        raw = np.asarray(result['raw_coordinates'], dtype=np.float64)
        if raw.size == 0:
            raw = np.empty((0, n_atoms, 3))
        if raw.ndim != 3 or raw.shape[1:] != (n_atoms, 3) or len(raw) > count:
            raise ValueError('raw coordinates must follow the input all-H atom order')
        raw_padded = np.full_like(output, np.nan)
        raw_padded[:len(raw)] = raw
        result['raw_coordinates'] = raw_padded
        result['raw_present_mask'] = np.arange(count) < len(raw)
    return result


def coordinate_identity(molecule, coordinates):
    """Measure coordinate finiteness and retention of specified stereochemistry."""
    from conformer_fidelity.chemistry.stereo import stereochemistry_signature, stereochemistry_retention

    reference = stereochemistry_signature(molecule)
    rows = []
    for i, xyz in enumerate(coordinates):
        finite = bool(np.isfinite(xyz).all())
        stereo = False
        failure = 'nonfinite_coordinates' if not finite else None
        if finite:
            observed = stereochemistry_signature(molecule, positions=xyz, from_geometry=True)
            stereo = bool(stereochemistry_retention(reference, observed).all_retained)
            if not stereo:
                failure = 'specified_stereochemistry_changed'
        rows.append(dict(native_index=i, finite=finite, specified_stereo_retained=stereo,
            identity_passed=finite and stereo, failure=failure))
    return rows
