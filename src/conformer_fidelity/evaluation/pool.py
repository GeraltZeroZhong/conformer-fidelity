"""Common fixed-output scoring: finite, original-eligible and jointly qualified."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from conformer_fidelity.chemistry import graph_context, original_eligibility
from .qualification import configuration, metrics, observe
from .rmsd import pairwise_ensemble_rmsd

FIELDS = (
    "qhit_1", "qhit_2", "qualified_best_rmsd", "output_fraction",
    "qualified_output_fraction", "geometry_fraction", "eligible_fraction",
    "finite_hit_1", "finite_hit_2", "finite_best_rmsd", "original_hit_1",
    "original_hit_2", "original_best_rmsd", "original_hit_loss_1",
    "original_hit_loss_2", "geometry_hit_loss_1", "geometry_hit_loss_2",
)


def target_distances(observation, target):
    """Measure all finite coordinates, including topology/stereo-ineligible ones."""
    context = observation['context']
    xyz = np.asarray(observation['coordinates'])
    target = np.asarray(target)
    if target.shape != (len(context['heavy']), 3) or not np.isfinite(target).all():
        raise ValueError('target must be finite and in original Z>1 heavy order')
    finite = np.isfinite(xyz).all(axis=(1, 2))
    valid = np.flatnonzero(finite)
    distance = np.full(len(xyz), 100.)
    if len(valid):
        distance[valid] = pairwise_ensemble_rmsd(
            torch.as_tensor(xyz[valid][:, context['heavy']], dtype=torch.float32).double(),
            torch.as_tensor(target, dtype=torch.float32)[None].double(),
            permutations=context['permutations']).numpy()[:, 0]
    if not np.isfinite(distance).all():
        raise ValueError('nonfinite RMSD for finite candidate coordinates')
    return distance


def _score_indices(observation, selected, distance, denominator):
    selected = np.asarray(selected, dtype=np.int64)
    xyz = np.asarray(observation['coordinates'])
    finite = np.isfinite(xyz).all(axis=(1, 2))
    original = np.asarray(observation['eligible'], dtype=bool) & finite
    geometry = observation['geometry']
    statuses = observation.get('statuses', [None]*len(xyz))
    value = metrics(distance, selected, geometry, original, statuses, denominator=denominator)
    if 'statuses' not in observation:
        value.pop('converged_fraction')
        value['convergence_status'] = 'not_comparable_native_search'
    for prefix, mask in (('finite', finite), ('original', original)):
        kept = selected[mask[selected]]
        for threshold in (1, 2):
            value[f'{prefix}_hit_{threshold}'] = float(bool(len(kept) and (distance[kept] <= threshold).any()))
        value[f'{prefix}_best_rmsd'] = float(distance[kept].min()) if len(kept) else 100.
        value[f'{prefix}_count'] = len(kept)
    for threshold in (1, 2):
        value[f'original_hit_loss_{threshold}'] = value[f'finite_hit_{threshold}']-value[f'original_hit_{threshold}']
        value[f'geometry_hit_loss_{threshold}'] = value[f'original_hit_{threshold}']-value[f'qhit_{threshold}']
    failures = observation.get('failures', [None]*len(xyz))
    reasons = Counter(str(failures[int(i)] or ('nonfinite_positions' if not finite[i]
        else 'original_eligibility_failure')) for i in selected if not original[i])
    failed_tests = Counter(name for i in selected if original[i]
        for name, passed in geometry[int(i)].get('tests', {}).items() if not passed)
    value.update(output_count=len(selected), output_fraction=len(selected)/denominator,
        qualified_output_fraction=value['qualified_count']/len(selected) if len(selected) else 0.,
        nonfinite_count=int((~finite[selected]).sum()),
        original_failure_reasons=dict(reasons), geometry_test_failures=dict(failed_tests),
        geometry_observation_errors=sum(bool(geometry[int(i)].get('error')) for i in selected if original[i]))
    return value


def observe_pool(smiles, coordinates, present_mask=None, *, geometry_config=None):
    """Observe fixed all-H input-order slots without accessing a target.

    Coordinates have shape [N, all-H atoms, 3]. A false present mask excludes
    that slot even if its placeholder happens to be finite. A returned but
    nonfinite slot stays present and receives failure metrics during scoring.
    """
    context = graph_context({'smiles': smiles})
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (context['molecule'].GetNumAtoms(), 3):
        raise ValueError('coordinates must be [N, input all-H atoms, 3]')
    present = np.ones(len(xyz), dtype=bool) if present_mask is None else np.asarray(present_mask)
    if present.shape != (len(xyz),) or present.dtype != np.bool_:
        raise ValueError('present_mask must be a boolean vector aligned with slots')
    xyz = xyz.copy()
    xyz[~present] = np.nan
    failures = [original_eligibility(context, candidate) if present[i] else 'not_returned'
                for i, candidate in enumerate(xyz)]
    eligible = np.array([reason is None for reason in failures], dtype=bool)
    config = configuration() if geometry_config is None else geometry_config
    geometry = observe(context['molecule'], xyz, config=config, eligible=eligible)
    return dict(context=context, coordinates=xyz, present_mask=present.copy(),
                eligible=eligible, failures=failures, geometry=geometry)


def score_fixed_selection(observation, selected, target, *, denominator=20, target_distance=None):
    """Score a fixed selection against a finite heavy-only target.

    A cached target_distance, when supplied, must belong to this exact same
    observation and target. It contains distances for all finite slots.
    """
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator < 1:
        raise ValueError('denominator must be a positive integer')
    distance = (target_distances(observation, target) if target_distance is None
                else np.asarray(target_distance, dtype=np.float64))
    if distance.shape != (len(observation['coordinates']),) or not np.isfinite(distance).all():
        raise ValueError('aligned finite full-pool distances required')
    return _score_indices(observation, selected, distance, denominator)


def score_pool(smiles, coordinates, target_coordinates, present_mask=None, denominator=20):
    """Evaluate one fixed conformer pool under the common three qualification levels.

    Candidate coordinates must use the all-H atom order of input_molecule(smiles).
    The target is either [all-H atoms, 3] in that same order, or [heavy atoms, 3]
    in the input graph's increasing Z>1 atom-index order. Every supplied target
    coordinate must be finite. Use a heavy-only target array when hydrogen
    coordinates are unavailable.

    Candidate and target RMSD coordinates are quantized to float32 then evaluated
    in float64. Alignment uses proper rotations and the same stereo-preserving
    heavy-atom permutations. Output and eligibility fractions keep the requested
    denominator (default 20); qualified_output_fraction uses returned slots.
    An empty qualification level receives a best-RMSD sentinel of 100 Angstrom.
    Qualification applies the six ligand-only geometry checks documented by
    the evaluation module.
    """
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator < 1:
        raise ValueError('denominator must be a positive integer')
    observation = observe_pool(smiles, coordinates, present_mask)
    context = observation['context']
    target = np.asarray(target_coordinates, dtype=np.float64)
    all_shape = (context['molecule'].GetNumAtoms(), 3)
    heavy_shape = (len(context['heavy']), 3)
    if target.shape not in (all_shape, heavy_shape) or not np.isfinite(target).all():
        raise ValueError('target must be finite all-H or explicit input-order heavy coordinates')
    if len(observation['coordinates']) > denominator:
        raise ValueError('pool exceeds requested denominator; select before scoring')
    if target.shape == all_shape:
        target = target[context['heavy']]
    distance = target_distances(observation, target)
    selected = np.flatnonzero(observation['present_mask'])
    value = score_fixed_selection(observation, selected, target,
                                  denominator=denominator, target_distance=distance)
    finite = observation['present_mask'] & np.isfinite(observation['coordinates']).all(axis=(1, 2))
    qualified = observation['eligible'] & np.array([r['passed'] for r in observation['geometry']], dtype=bool)
    value.update(requested_denominator=denominator, target_distance=distance.tolist(),
                 present_mask=observation['present_mask'].tolist(), finite_mask=finite.tolist(),
                 original_mask=observation['eligible'].tolist(), qualified_mask=qualified.tolist(),
                 original_failures=observation['failures'], geometry=observation['geometry'])
    return value
