"""Target-blind ETKDGv3 and fixed-ID MMFF94s relaxation.

Embedding uses single-threaded ETKDGv3 with enforced chirality. Relaxation uses
MMFF94s and RDKit's default convergence tolerances. The caller supplies the
conformer count and maximum iteration count.
"""
from __future__ import annotations

import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from ._common import fixed_slots, graph_record, input_molecule, validate_request


def generate(smiles, *, count=20, seed=0):
    """Generate one ETKDGv3 pool in a single embedding call."""
    validate_request(count, seed)
    if seed >= 2**31:
        raise ValueError('ETKDG randomSeed must be within signed int32 range')
    started = time.perf_counter()
    molecule = input_molecule(smiles)
    copied = Chem.Mol(molecule)
    copied.RemoveAllConformers()
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.clearConfs = True
    parameters.enforceChirality = True
    conformer_ids = tuple(int(value) for value in
        AllChem.EmbedMultipleConfs(copied, numConfs=count, params=parameters))
    coordinates = np.stack([np.asarray(copied.GetConformer(value).GetPositions())
        for value in conformer_ids]) if conformer_ids else np.empty((0, copied.GetNumAtoms(), 3))
    result = dict(method='etkdg-v3', smiles=smiles, seed=seed, seed_controlled=True,
        molecule=molecule, coordinates=np.asarray(coordinates, dtype=np.float32), conformer_ids=conformer_ids,
        failure=None, rdkit=rdBase.rdkitVersion, num_threads=1, external_optimization=False,
        external_selection=False, elapsed_seconds=time.perf_counter()-started)
    return fixed_slots(result, count)


def relax(smiles, coordinates, *, max_iterations=200, present_mask=None):
    """Retain fixed slots; status 1 coordinates survive, failures become NaN.

    Every finite, present slot receives unconstrained MMFF94s minimization.
    ``present_mask`` records returned native slots. Optimizer status and energies
    are stored for each attempted slot; chemical qualification is a separate step.
    """
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    molecule = input_molecule(smiles)
    original_graph = graph_record(molecule)
    xyz = np.asarray(coordinates)
    if xyz.ndim != 3 or xyz.shape[1:] != (molecule.GetNumAtoms(), 3):
        raise ValueError('coordinates must have shape (slots, input all-H atoms, 3)')
    count = len(xyz)
    present = np.ones(count, dtype=bool) if present_mask is None else np.asarray(present_mask, dtype=bool)
    if present.shape != (count,):
        raise ValueError('present_mask must have one element per fixed input slot')
    output = np.full(xyz.shape, np.nan, dtype=np.float64)
    statuses, failures = [None]*count, [None]*count
    attempted, seconds = [False]*count, [0.]*count
    initial_energy, final_energy = [None]*count, [None]*count
    setup_start = time.perf_counter()
    try:
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant='MMFF94s')
        if properties is None:
            raise ValueError('MMFF94s parameters unavailable')
        setup_failure = None
    except (ValueError, RuntimeError) as exc:
        properties = None
        setup_failure = f'{type(exc).__name__}: {exc}'
    setup_seconds = time.perf_counter()-setup_start
    for slot in range(count):
        if not present[slot]:
            failures[slot] = 'native_missing_slot'
            continue
        if not np.isfinite(xyz[slot]).all():
            failures[slot] = 'native_nonfinite_coordinates'
            continue
        if setup_failure:
            failures[slot] = 'mmff_setup_failed: '+setup_failure
            continue
        tick = time.perf_counter()
        attempted[slot] = True
        try:
            molecule.RemoveAllConformers()
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            conformer.Set3D(True)
            for index, point in enumerate(xyz[slot]):
                conformer.SetAtomPosition(index, tuple(map(float, point)))
            molecule.AddConformer(conformer)
            forcefield = AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0)
            if forcefield is None:
                raise ValueError('MMFF94s forcefield unavailable')
            forcefield.Initialize()
            energy = float(forcefield.CalcEnergy())
            initial_energy[slot] = energy if np.isfinite(energy) else None
            status = int(forcefield.Minimize(maxIts=max_iterations))
            statuses[slot] = status
            result = np.asarray(molecule.GetConformer().GetPositions())
            if status not in (0, 1):
                raise ValueError(f'MMFF failure status {status}')
            if not np.isfinite(result).all():
                raise ValueError('nonfinite MMFF coordinates')
            energy = float(forcefield.CalcEnergy())
            final_energy[slot] = energy if np.isfinite(energy) else None
            output[slot] = result
        except (ValueError, RuntimeError) as exc:
            failures[slot] = f'{type(exc).__name__}: {exc}'
        seconds[slot] = time.perf_counter()-tick
    return dict(coordinates=output, present_mask=present.copy(), molecule=molecule,
        smiles=smiles, requested=count, target_access=False, external_selection=False,
        optimizer_status=statuses, optimizer_attempted=attempted,
        optimizer_failure=failures, optimizer_seconds=seconds,
        optimizer_initial_energy=initial_energy, optimizer_final_energy=final_energy,
        optimizer_setup_seconds=setup_seconds, optimizer_setup_failure=setup_failure,
        forcefield='MMFF94s', iterations=max_iterations,
        fixed_input_slot_semantics='present_mask records the native returned slots',
        output_finite_count=int(np.isfinite(output).all(axis=(1, 2)).sum()),
        graph=original_graph)
