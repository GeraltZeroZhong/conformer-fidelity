"""Deterministic target-blind selection from fixed pool distances and energies.

Six rules combine energy ordering and geometric diversity with fixed settings.
"""
from __future__ import annotations

import numpy as np

METHODS = (
    'facility', 'energy_facility', 'energy', 'kcenter',
    'energy_rmsd05', 'energy_rmsd10',
)


def _mask(value, size, name):
    mask = np.ones(size, dtype=bool) if value is None else np.asarray(value)
    if mask.shape != (size,) or mask.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean vector aligned with pool slots')
    return mask


def _midrank(values):
    if len(values) < 2:
        return np.zeros(len(values))
    return ((values[:, None] > values).sum(1)
            + .5 * ((values[:, None] == values).sum(1) - 1)) / (len(values) - 1)


def select_indices(distance, energies, method='facility', maximum=20,
                   present_mask=None, eligible_mask=None):
    """Select existing pool slot indices using energy and geometric diversity.

    Inputs are pool-to-pool heavy RMSDs [N,N] and physical energies [N]. Masks
    retain the original N-slot indexing. Present, chemically eligible slots
    can be selected; masked-out rows may contain NaNs. Eligibility is determined
    from each candidate's chemistry and coordinates before target scoring.

    Facility greedily maximizes equal-demand nearest-representative improvement.
    Energy-facility uses the fixed 0.5/0.5 mixture of energy and improvement
    midranks. Ties use original slot order. K-center starts at the lowest energy
    (energy ties by slot order), then maximizes minimum selected-set RMSD.
    Energy-RMSD walks stable energy order and admits only candidates strictly
    farther than 0.5 or 1.0 Angstrom from all selected candidates. It can return
    fewer than maximum even if additional eligible candidates remain.

    The output limit is 0..20 geometric representatives of the supplied pool.
    """
    if method not in METHODS:
        raise ValueError(f'unknown selection method: {method}')
    if (not isinstance(maximum, (int, np.integer)) or isinstance(maximum, (bool, np.bool_))
            or not 0 <= maximum <= 20):
        raise ValueError('maximum must be an integer from 0 to 20')
    energies = np.asarray(energies, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if energies.ndim != 1 or distance.shape != (len(energies), len(energies)):
        raise ValueError('aligned one-dimensional energies and square pool distances required')
    valid = np.flatnonzero(_mask(present_mask, len(energies), 'present_mask')
                          & _mask(eligible_mask, len(energies), 'eligible_mask'))
    distances = distance[np.ix_(valid, valid)]
    energy = energies[valid]
    if not np.isfinite(distances).all() or not np.isfinite(energy).all():
        raise ValueError('all present eligible features must be finite')
    if (distances < 0).any() or not np.allclose(distances, distances.T, rtol=0, atol=1e-8):
        raise ValueError('pool RMSDs must be nonnegative and symmetric')
    if not np.allclose(np.diag(distances), 0, rtol=0, atol=1e-8):
        raise ValueError('pool RMSD diagonal must be zero')
    limit = min(maximum, len(valid))
    if limit == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(energy, kind='stable')
    if method == 'energy':
        return valid[order[:limit]]
    if method in ('facility', 'energy_facility'):
        weight = 1. if method == 'facility' else .5
        selected = []
        current = np.full(len(valid), distances.max())
        energy_rank = _midrank(-energy) if weight < 1 else None
        while len(selected) < limit:
            score = np.zeros(len(valid))
            if weight < 1:
                score += (1 - weight) * energy_rank
            gain = np.maximum(current[:, None] - distances, 0).mean(0)
            score += weight * _midrank(gain)
            score[selected] = -np.inf
            chosen = int(np.argmax(score))
            selected.append(chosen)
            current = np.minimum(current, distances[:, chosen])
        return valid[np.asarray(selected, dtype=np.int64)]
    selected = [int(order[0])]
    if method == 'kcenter':
        nearest = distances[:, selected[0]].copy()
        while len(selected) < limit:
            nearest[selected] = -np.inf
            chosen = int(np.argmax(nearest))
            selected.append(chosen)
            nearest = np.minimum(nearest, distances[:, chosen])
    else:
        radius = .5 if method == 'energy_rmsd05' else 1.
        for index in order[1:]:
            if len(selected) >= limit:
                break
            if np.all(distances[int(index), selected] > radius):
                selected.append(int(index))
    return valid[np.asarray(selected, dtype=np.int64)]
