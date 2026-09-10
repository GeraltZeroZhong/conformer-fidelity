"""Fixed-pool chemical eligibility, RMSD and single-point energy."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from conformer_fidelity.chemistry import graph_context, original_eligibility
from .rules import _mask, select_indices


def _single_energies(molecule, coordinates):
    """Evaluate MMFF94s single-point energies at the supplied coordinates."""
    work = Chem.Mol(molecule)
    work.RemoveAllConformers()
    work.AddConformer(Chem.Conformer(work.GetNumAtoms()), assignId=True)
    props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant='MMFF94s')
    if props is None:
        raise ValueError('MMFF94s parameters unavailable')
    energies = []
    for xyz in coordinates:
        for i, pos in enumerate(xyz):
            work.GetConformer().SetAtomPosition(i, tuple(map(float, pos)))
        ff = AllChem.MMFFGetMoleculeForceField(work, props)
        if ff is None:
            raise ValueError('MMFF94s force field unavailable')
        ff.Initialize()
        energies.append(float(ff.CalcEnergy()))
    if not np.isfinite(energies).all():
        raise ValueError('nonfinite single-point energy')
    return np.asarray(energies, dtype=np.float64)


def pool_geometry(smiles, coordinates, present_mask=None):
    """Compute pool features while retaining [N, all-H atoms, 3] slot indexing.

    Atom order follows input_molecule(smiles). Present, chemically eligible
    coordinates are quantized to float32, then evaluated in float64 for both
    RMSD and single-point energy. Masked feature entries contain NaN.
    Pass both returned masks to select_indices.
    """
    import torch
    from conformer_fidelity.evaluation.rmsd import pairwise_ensemble_rmsd

    context = graph_context({'smiles': smiles})
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (context['molecule'].GetNumAtoms(), 3):
        raise ValueError('coordinates must be [N, input all-H atoms, 3]')
    present = _mask(present_mask, len(xyz), 'present_mask')
    failures = [original_eligibility(context, value) if present[i] else 'not_returned'
                for i, value in enumerate(xyz)]
    eligible = np.array([reason is None for reason in failures], dtype=bool)
    valid = np.flatnonzero(present & eligible)
    distances = np.full((len(xyz), len(xyz)), np.nan)
    energies = np.full(len(xyz), np.nan)
    if len(valid):
        quantized = xyz[valid].astype(np.float32).astype(np.float64)
        positions = torch.as_tensor(quantized[:, context['heavy']], dtype=torch.float64)
        d = pairwise_ensemble_rmsd(positions, positions,
                                  permutations=context['permutations']).numpy()
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0.)
        distances[np.ix_(valid, valid)] = d
        energies[valid] = _single_energies(context['molecule'], quantized)
    return dict(distance=distances, energies=energies, present_mask=present.copy(),
                eligible_mask=eligible, failures=failures, target_access=False)


def select_conformers(smiles, coordinates, method='facility', maximum=20, present_mask=None):
    """Return selected pool indices and metadata using a fixed geometric rule.

    Repeated calls on the same ordered pool produce the same selection.
    Coordinates remain unchanged, and generation and relaxation run separately.
    """
    features = pool_geometry(smiles, coordinates, present_mask)
    indices = select_indices(features['distance'], features['energies'], method, maximum,
                             features['present_mask'], features['eligible_mask'])
    return dict(indices=indices, method=method, maximum=int(maximum),
                selected_count=len(indices), input_count=len(features['energies']),
                eligible_count=int(features['eligible_mask'].sum()),
                present_mask=features['present_mask'], eligible_mask=features['eligible_mask'],
                target_access=False, coordinates_changed=False, refill=False)
