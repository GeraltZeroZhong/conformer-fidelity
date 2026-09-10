"""Chunked symmetry-aware heavy-atom Kabsch RMSD, without reflection."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


DEFAULT_PERMUTATION_CHUNK_SIZE = 256
DEFAULT_PAIR_CHUNK_SIZE = 32


def _batched_kabsch_rmsd(
    probe: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Kabsch RMSD over broadcast-compatible batches ending in ``(atoms, 3)``."""
    if probe.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError("probe and reference coordinates must end in (atoms, 3)")
    if probe.shape[-2] != reference.shape[-2]:
        raise ValueError("probe and reference must contain the same number of atoms")
    atoms = probe.shape[-2]
    if weights is None:
        weights = torch.ones(atoms, dtype=probe.dtype, device=probe.device)
    else:
        weights = torch.as_tensor(weights, dtype=probe.dtype, device=probe.device)
    weights = weights / weights.sum()
    weight_shape = (1,) * (max(probe.ndim, reference.ndim) - 2) + (atoms, 1)
    broadcast_weights = weights.reshape(weight_shape)
    probe_centered = probe - (broadcast_weights * probe).sum(dim=-2, keepdim=True)
    reference_centered = reference - (broadcast_weights * reference).sum(dim=-2, keepdim=True)
    covariance = (broadcast_weights * probe_centered).transpose(-2, -1) @ reference_centered
    u, _, vh = torch.linalg.svd(covariance)
    correction = (
        torch.eye(3, dtype=probe.dtype, device=probe.device).expand(covariance.shape).clone()
    )
    correction[..., -1, -1] = torch.sign(torch.linalg.det(u @ vh))
    rotation = u @ correction @ vh
    aligned = probe_centered @ rotation
    squared = (
        (aligned - reference_centered).square().sum(dim=-1) * broadcast_weights.squeeze(-1)
    ).sum(dim=-1)
    return torch.sqrt(torch.clamp_min(squared, 0.0))


def _masked_coordinates(
    probe: torch.Tensor,
    reference: torch.Tensor,
    atom_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if atom_mask is None:
        return probe, reference
    mask = torch.as_tensor(atom_mask, dtype=torch.bool, device=probe.device)
    return probe[..., mask, :], reference[..., mask, :]


def _permutation_chunks(
    permutations: Sequence[Sequence[int]],
    *,
    atoms: int,
    device: torch.device,
    chunk_size: int,
) -> Iterable[torch.Tensor]:
    if chunk_size < 1:
        raise ValueError("permutation_chunk_size must be positive")
    for left in range(0, len(permutations), chunk_size):
        chunk = torch.as_tensor(
            permutations[left : left + chunk_size],
            dtype=torch.long,
            device=device,
        )
        if chunk.ndim != 2 or chunk.shape[1] != atoms:
            raise ValueError("each atom permutation must match the masked atom count")
        yield chunk


def kabsch_rmsd(
    probe: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rigid-alignment RMSD for corresponding points, without reflection."""
    probe = torch.as_tensor(probe)
    reference = torch.as_tensor(reference, device=probe.device, dtype=probe.dtype)
    if probe.shape != reference.shape or probe.shape[-1] != 3:
        raise ValueError("probe and reference must have the same (..., atoms, 3) shape")
    if probe.ndim != 2:
        raise ValueError("kabsch_rmsd currently expects a single (atoms, 3) pair")
    return _batched_kabsch_rmsd(probe, reference, weights)


def symmetry_aware_rmsd(
    probe: torch.Tensor,
    reference: torch.Tensor,
    permutations: Sequence[Sequence[int]] | None = None,
    atom_mask: torch.Tensor | None = None,
    *,
    permutation_chunk_size: int = DEFAULT_PERMUTATION_CHUNK_SIZE,
) -> torch.Tensor:
    """Minimum heavy-atom Kabsch RMSD over bounded batches of atom permutations."""
    probe = torch.as_tensor(probe)
    reference = torch.as_tensor(reference, dtype=probe.dtype, device=probe.device)
    probe, reference = _masked_coordinates(probe, reference, atom_mask)
    if permutations is None or len(permutations) == 0:
        return kabsch_rmsd(probe, reference)
    minimum = torch.full((), torch.inf, dtype=probe.dtype, device=probe.device)
    for permutation in _permutation_chunks(
        permutations,
        atoms=probe.shape[-2],
        device=probe.device,
        chunk_size=permutation_chunk_size,
    ):
        values = _batched_kabsch_rmsd(probe[permutation], reference)
        minimum = torch.minimum(minimum, values.min())
    return minimum


def pairwise_ensemble_rmsd(
    generated: torch.Tensor,
    references: torch.Tensor,
    permutations: Sequence[Sequence[int]] | None = None,
    atom_mask: torch.Tensor | None = None,
    *,
    permutation_chunk_size: int = DEFAULT_PERMUTATION_CHUNK_SIZE,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> torch.Tensor:
    """Return a chunked ``(n_generated, n_reference)`` symmetry-aware RMSD matrix."""
    generated = torch.as_tensor(generated)
    references = torch.as_tensor(references, dtype=generated.dtype, device=generated.device)
    generated, references = _masked_coordinates(generated, references, atom_mask)
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive")
    generated_count = generated.shape[0]
    reference_count = references.shape[0]
    pair_count = generated_count * reference_count
    pair_indices = torch.arange(pair_count, device=generated.device)

    if permutations is None or len(permutations) == 0:
        pair_values = []
        for left in range(0, pair_count, pair_chunk_size):
            indices = pair_indices[left : left + pair_chunk_size]
            probe = generated[torch.div(indices, reference_count, rounding_mode="floor")]
            reference = references[indices % reference_count]
            pair_values.append(_batched_kabsch_rmsd(probe, reference))
        flat = torch.cat(pair_values) if pair_values else generated.new_empty((0,))
        return flat.reshape(generated_count, reference_count)

    minimum = torch.full((pair_count,), torch.inf, dtype=generated.dtype, device=generated.device)
    for permutation in _permutation_chunks(
        permutations,
        atoms=generated.shape[-2],
        device=generated.device,
        chunk_size=permutation_chunk_size,
    ):
        pair_values = []
        for left in range(0, pair_count, pair_chunk_size):
            indices = pair_indices[left : left + pair_chunk_size]
            probe = generated[torch.div(indices, reference_count, rounding_mode="floor")]
            reference = references[indices % reference_count]
            permuted = probe[:, permutation]
            values = _batched_kabsch_rmsd(permuted, reference[:, None])
            pair_values.append(values.min(dim=1).values)
        if pair_values:
            minimum = torch.minimum(minimum, torch.cat(pair_values))
    return minimum.reshape(generated_count, reference_count)
