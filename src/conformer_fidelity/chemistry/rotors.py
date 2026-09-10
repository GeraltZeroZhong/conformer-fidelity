"""Bridge-rotor bookkeeping and torsion extraction; no training dependencies."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Rotor:
    """One oriented bridge rotor and the atoms moved by its positive rotation."""

    anchor: int
    pivot: int
    left_reference: int
    right_reference: int
    rotating_atoms: tuple[int, ...]
    bond_index: int

    @property
    def dihedral(self) -> tuple[int, int, int, int]:
        return (self.left_reference, self.anchor, self.pivot, self.right_reference)



def _component_without_bond(
    adjacency: tuple[tuple[int, ...], ...], start: int, blocked: frozenset[int]
) -> tuple[int, ...]:
    """Return the sorted component reachable without crossing ``blocked``."""
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if frozenset((atom, neighbor)) == blocked or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return tuple(sorted(seen))


def _heavy_atoms(mol: Chem.Mol, atoms: Iterable[int]) -> tuple[int, ...]:
    return tuple(index for index in atoms if mol.GetAtomWithIdx(index).GetAtomicNum() > 1)


def _strict_rotatable_pairs(
    mol: Chem.Mol, non_strict_pairs: tuple[tuple[int, int], ...]
) -> set[frozenset[int]]:
    """Recover RDKit's per-bond strict decisions from its public count API.

    RDKit exposes atom pairs for its non-strict SMARTS, while the strict API
    exposes only a count.  Replacing one candidate by a zero-order bond keeps
    atom degrees and every other bond unchanged but removes that candidate
    from the strict descriptor.  A one-count decrease therefore identifies a
    bond accepted by RDKit's own strict definition (and rejects amide/ester
    bonds without maintaining a second copy of RDKit's chemistry rules).
    """
    options = rdMolDescriptors.NumRotatableBondsOptions.Strict
    baseline = rdMolDescriptors.CalcNumRotatableBonds(mol, options)
    accepted: set[frozenset[int]] = set()
    for first, second in non_strict_pairs:
        editable = Chem.RWMol(mol)
        bond = editable.GetBondBetweenAtoms(first, second)
        if bond is None:
            continue
        bond.SetBondType(Chem.BondType.ZERO)
        perturbed = editable.GetMol()
        count = rdMolDescriptors.CalcNumRotatableBonds(perturbed, options)
        if count < baseline:
            accepted.add(frozenset((first, second)))
    return accepted


def _reference_neighbor(mol: Chem.Mol, atom: int, other_axis_atom: int) -> int:
    candidates = sorted(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors()
        if neighbor.GetIdx() != other_axis_atom and neighbor.GetAtomicNum() > 1
    )
    if not candidates:
        raise ValueError(
            f"rotor endpoint {atom} has no heavy reference neighbor besides {other_axis_atom}"
        )
    return candidates[0]


def find_bridge_rotors(
    mol: Chem.Mol,
    *,
    strict: bool = True,
    min_heavy_atoms_per_side: int = 2,
) -> tuple[Rotor, ...]:
    """Find oriented, non-ring bridge rotors in an RDKit molecule.

    Candidate axes are single heavy-atom bonds outside rings.  Both components
    created by cutting the bond must contain at least
    ``min_heavy_atoms_per_side`` heavy atoms.  With ``strict=True`` (the
    default), RDKit's strict rotatable-bond definition additionally removes
    amide, ester, and other resonance-restricted axes.

    The component with fewer heavy atoms rotates.  Equal-size components are
    resolved by their sorted atom-index tuple.  ``rotating_atoms`` contains all
    atoms in that component, including explicit hydrogens and the pivot, but
    never the anchor.  This makes the orientation and mask deterministic.
    """
    if min_heavy_atoms_per_side < 2:
        raise ValueError("min_heavy_atoms_per_side must be at least 2")

    adjacency = tuple(
        tuple(sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors()))
        for atom in mol.GetAtoms()
    )
    non_strict_pairs = tuple(mol.GetSubstructMatches(Lipinski.RotatableBondSmarts))
    non_strict = {frozenset(pair) for pair in non_strict_pairs}
    strict_pairs = _strict_rotatable_pairs(mol, non_strict_pairs) if strict else non_strict

    rotors: list[Rotor] = []
    for bond in mol.GetBonds():
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        pair = frozenset((first, second))
        if (
            bond.GetBondType() != Chem.BondType.SINGLE
            or bond.GetIsAromatic()
            or bond.IsInRing()
            or pair not in strict_pairs
            or mol.GetAtomWithIdx(first).GetAtomicNum() <= 1
            or mol.GetAtomWithIdx(second).GetAtomicNum() <= 1
        ):
            continue

        first_side = _component_without_bond(adjacency, first, pair)
        second_side = _component_without_bond(adjacency, second, pair)
        # A non-ring edge should be a bridge, but checking the partition here
        # also handles unsanitized/query molecules without guessing.
        if len(first_side) + len(second_side) != mol.GetNumAtoms():
            continue
        first_heavy = _heavy_atoms(mol, first_side)
        second_heavy = _heavy_atoms(mol, second_side)
        if min(len(first_heavy), len(second_heavy)) < min_heavy_atoms_per_side:
            continue

        first_key = (len(first_heavy), first_side)
        second_key = (len(second_heavy), second_side)
        if first_key <= second_key:
            pivot, anchor = first, second
            rotating_atoms = first_side
        else:
            pivot, anchor = second, first
            rotating_atoms = second_side

        rotors.append(
            Rotor(
                anchor=anchor,
                pivot=pivot,
                left_reference=_reference_neighbor(mol, anchor, pivot),
                right_reference=_reference_neighbor(mol, pivot, anchor),
                rotating_atoms=rotating_atoms,
                bond_index=bond.GetIdx(),
            )
        )

    return tuple(sorted(rotors, key=lambda rotor: rotor.bond_index))


def _check_positions(positions: torch.Tensor) -> None:
    import torch

    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim < 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (..., atoms, 3)")
    if not positions.is_floating_point():
        raise TypeError("positions must use a floating-point dtype")


def _unit_vector(vector: torch.Tensor) -> torch.Tensor:
    import torch

    minimum = torch.finfo(vector.dtype).eps
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(minimum)


def dihedral_angle(positions: torch.Tensor, atoms: tuple[int, int, int, int]) -> torch.Tensor:
    """Return the signed dihedral for ``(left, anchor, pivot, right)``.

    Positive rotation follows the right-hand rule around the axis from anchor
    to pivot.  Leading batch dimensions are preserved.
    """
    import torch

    _check_positions(positions)
    left, anchor, pivot, right = atoms
    p0 = positions[..., left, :]
    p1 = positions[..., anchor, :]
    p2 = positions[..., pivot, :]
    p3 = positions[..., right, :]

    axis = _unit_vector(p2 - p1)
    first = p0 - p1
    second = p3 - p2
    first = first - (first * axis).sum(dim=-1, keepdim=True) * axis
    second = second - (second * axis).sum(dim=-1, keepdim=True) * axis
    cosine = (first * second).sum(dim=-1)
    sine = (torch.linalg.cross(axis, first, dim=-1) * second).sum(dim=-1)
    return torch.atan2(sine, cosine)


def extract_torsions(positions: torch.Tensor, rotors: Sequence[Rotor]) -> torch.Tensor:
    """Extract all rotor dihedrals as a tensor with shape ``(..., rotors)``."""
    import torch

    _check_positions(positions)
    if not rotors:
        return positions.new_empty(positions.shape[:-2] + (0,))
    return torch.stack([dihedral_angle(positions, rotor.dihedral) for rotor in rotors], dim=-1)
