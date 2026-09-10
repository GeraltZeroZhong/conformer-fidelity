"""Six ligand-only geometric checks and fixed-output qualification metrics."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import time

import numpy as np
import posebusters
from posebusters.modules.distance_geometry import check_geometry
from posebusters.modules.flatness import check_flatness
import yaml

from conformer_fidelity.chemistry.stereo import _molecule_with_positions


def configuration():
    path = Path(posebusters.__file__).parent / 'config/mol.yml'
    original = yaml.safe_load(path.read_text())
    modules = [m for m in original['modules'] if m['function'] in ('distance_geometry', 'flatness')]
    if len(modules) != 4 or modules[0]['parameters']['threshold_bad_bond_length'] != .25:
        raise ValueError('installed PoseBusters geometry defaults differ from supported configuration')
    return dict(posebusters=importlib.metadata.version('posebusters'), modules=modules,
                additional_energy_sampling=False, target_input=False,
                interpretation='geometry_subset_not_full_PoseBusters_or_thermodynamic_validity')


def observe(molecule, coordinates, *, config, eligible=None, check=lambda: None):
    coords = np.asarray(coordinates)
    eligible = np.ones(len(coords), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if len(eligible) != len(coords):
        raise ValueError('candidate eligibility length mismatch')
    output = []
    for i, xyz in enumerate(coords):
        check()
        tick = time.perf_counter()
        tests, error = {}, None
        if eligible[i] and np.isfinite(xyz).all():
            try:
                mol = _molecule_with_positions(molecule, xyz.astype(np.float32))
                for module in config['modules']:
                    function = {'distance_geometry': check_geometry, 'flatness': check_flatness}[module['function']]
                    result = function(mol, **module['parameters'])['results']
                    for key in module['chosen_binary_test_output']:
                        value = result.get(key)
                        # Missing/NaN results are failures, never truthy NaNs.
                        tests[module['name'] + '/' + key] = isinstance(value, (bool, np.bool_)) and bool(value)
            except (ValueError, RuntimeError, KeyError, AssertionError, np.linalg.LinAlgError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)
        else:
            error = 'original_eligibility_failure_or_nonfinite'
        passed = eligible[i] and error is None and len(tests) == 6 and all(tests.values())
        output.append(dict(index=i, passed=bool(passed), tests=tests, error=error,
                           seconds=time.perf_counter() - tick))
    return output


def metrics(distance, selected, geometry, eligible, statuses, *, denominator=20):
    """Same candidate must pass geometry and hit. Missing requests stay failures."""
    indices = np.asarray(selected, dtype=np.int64)
    d = np.asarray(distance, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if len(indices) > denominator or len(set(indices.tolist())) != len(indices):
        raise ValueError('invalid fixed selection or denominator')
    if len(indices) and ((indices < 0).any() or (indices >= len(d)).any()):
        raise ValueError('selection outside original pool')
    kept = indices[mask[indices]]
    qualified = np.array([i for i in kept if geometry[int(i)]['passed']], dtype=np.int64)
    raw = d[kept]
    quality = d[qualified]
    return dict(
        qhit_1=float(bool(len(quality) and (quality <= 1.).any())),
        raw_hit_1=float(bool(len(raw) and (raw <= 1.).any())),
        qhit_2=float(bool(len(quality) and (quality <= 2.).any())),
        raw_hit_2=float(bool(len(raw) and (raw <= 2.).any())),
        best_rmsd=float(raw.min()) if len(raw) else 100.,
        qualified_best_rmsd=float(quality.min()) if len(quality) else 100.,
        geometry_fraction=len(qualified) / denominator,
        eligible_fraction=len(kept) / denominator,
        converged_fraction=sum(statuses[int(i)] == 0 for i in kept) / denominator,
        eligible_count=len(kept), qualified_count=len(qualified), missing=denominator - len(kept),
        geometric_failed=len(kept) - len(qualified),
        geometry_seconds=sum(geometry[int(i)]['seconds'] for i in indices),
        raw_hit_count=int((raw <= 1.).sum()), qualified_hit_count=int((quality <= 1.).sum()))
