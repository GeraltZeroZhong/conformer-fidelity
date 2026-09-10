"""Stereochemistry-preserving heavy-atom graph automorphisms."""
from __future__ import annotations

from rdkit import Chem


def heavy_atom_automorphisms(
    molecule: Chem.Mol, *, max_automorphisms: int = 128
) -> tuple[tuple[int, ...], ...]:
    """Enumerate stereochemistry-preserving automorphisms of the heavy graph.

    Returned tuples gather the coordinate axis: ``permuted = positions[mapping]``.
    They have the original molecule's atom count; explicit hydrogens are fixed,
    while heavy atoms follow the RDKit automorphism.
    """
    if max_automorphisms < 1:
        raise ValueError("max_automorphisms must be positive")
    work = Chem.Mol(molecule)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    property_name = "_conformer_fidelity_original_atom_index"
    for atom in work.GetAtoms():
        atom.SetIntProp(property_name, atom.GetIdx())
    heavy = Chem.RemoveHs(work)
    original_indices = tuple(atom.GetIntProp(property_name) for atom in heavy.GetAtoms())
    identity = tuple(range(molecule.GetNumAtoms()))

    mappings = {identity}
    matches = heavy.GetSubstructMatches(
        heavy,
        uniquify=False,
        useChirality=True,
        maxMatches=max_automorphisms,
    )
    for match in matches:
        mapping = list(identity)
        for query_index, target_index in enumerate(match):
            mapping[original_indices[query_index]] = original_indices[target_index]
        mappings.add(tuple(mapping))
        if len(mappings) >= max_automorphisms:
            break
    return tuple(sorted(mappings, key=lambda mapping: (mapping != identity, mapping)))
