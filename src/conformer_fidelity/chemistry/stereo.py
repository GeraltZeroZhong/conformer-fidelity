"""Atom-indexed specified and independently observed three-dimensional stereo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class StereoSignature:
    chiral_centers: tuple[tuple[int, str], ...]
    double_bonds: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class StereoRetention:
    chirality_retained: int
    chirality_total: int
    ez_retained: int
    ez_total: int

    @property
    def chirality_rate(self) -> float:
        return self.chirality_retained / self.chirality_total if self.chirality_total else 1.0

    @property
    def ez_rate(self) -> float:
        return self.ez_retained / self.ez_total if self.ez_total else 1.0

    @property
    def all_retained(self) -> bool:
        return self.chirality_retained == self.chirality_total and self.ez_retained == self.ez_total


def _molecule_with_positions(molecule: Chem.Mol, positions: Tensor | np.ndarray) -> Chem.Mol:
    if hasattr(positions, 'detach'):
        positions = positions.detach().cpu().numpy()
    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError("positions must have shape [molecule atoms, 3]")
    copied = Chem.Mol(molecule)
    copied.RemoveAllConformers()
    conformer = Chem.Conformer(copied.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index, point in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, tuple(float(value) for value in point))
    copied.AddConformer(conformer, assignId=True)
    return copied


def stereochemistry_signature(
    molecule: Chem.Mol,
    *,
    positions: Tensor | np.ndarray | None = None,
    conf_id: int = -1,
    from_geometry: bool | None = None,
) -> StereoSignature:
    """Return atom-indexed R/S and bond-indexed E/Z labels.

    Supplying ``positions`` defaults to reassignment from those coordinates,
    so a mirrored output cannot inherit the template's original graph tags.
    Without positions, the molecule's specified graph stereochemistry is used.
    """
    copied = (
        Chem.Mol(molecule) if positions is None else _molecule_with_positions(molecule, positions)
    )
    use_geometry = positions is not None if from_geometry is None else bool(from_geometry)
    active_conf_id = 0 if positions is not None else conf_id
    if use_geometry:
        if copied.GetNumConformers() == 0:
            raise ValueError("geometry-based stereochemistry requires a conformer")
        Chem.AssignStereochemistryFrom3D(
            copied,
            confId=active_conf_id,
            replaceExistingTags=True,
        )
    else:
        Chem.AssignStereochemistry(copied, cleanIt=True, force=True)

    # Read E/Z first: FindMolChiralCenters currently normalizes RDKit's bond
    # enum from E/Z to the equivalent CIS/TRANS representation in-place.
    double_bonds: list[tuple[int, int, str]] = []
    for bond in copied.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            label = "E"
        elif stereo == Chem.BondStereo.STEREOZ:
            label = "Z"
        else:
            continue
        first, second = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        double_bonds.append((first, second, label))
    centers = tuple(
        sorted(
            (int(atom_index), label)
            for atom_index, label in Chem.FindMolChiralCenters(
                copied,
                includeUnassigned=False,
                useLegacyImplementation=False,
            )
            if label in {"R", "S"}
        )
    )
    return StereoSignature(centers, tuple(sorted(double_bonds)))


def chirality_signature(
    molecule: Chem.Mol,
    *,
    positions: Tensor | np.ndarray | None = None,
    conf_id: int = -1,
    from_geometry: bool | None = None,
) -> tuple[tuple[int, str], ...]:
    return stereochemistry_signature(
        molecule,
        positions=positions,
        conf_id=conf_id,
        from_geometry=from_geometry,
    ).chiral_centers


def ez_signature(
    molecule: Chem.Mol,
    *,
    positions: Tensor | np.ndarray | None = None,
    conf_id: int = -1,
    from_geometry: bool | None = None,
) -> tuple[tuple[int, int, str], ...]:
    return stereochemistry_signature(
        molecule,
        positions=positions,
        conf_id=conf_id,
        from_geometry=from_geometry,
    ).double_bonds


def stereochemistry_retention(
    reference: StereoSignature,
    candidate: StereoSignature,
) -> StereoRetention:
    """Compare labels defined by the reference fixed template."""
    candidate_centers = dict(candidate.chiral_centers)
    candidate_bonds = {(first, second): label for first, second, label in candidate.double_bonds}
    retained_centers = sum(
        candidate_centers.get(atom_index) == label for atom_index, label in reference.chiral_centers
    )
    retained_bonds = sum(
        candidate_bonds.get((first, second)) == label
        for first, second, label in reference.double_bonds
    )
    return StereoRetention(
        chirality_retained=int(retained_centers),
        chirality_total=len(reference.chiral_centers),
        ez_retained=int(retained_bonds),
        ez_total=len(reference.double_bonds),
    )


def independent_stereo(molecule, positions, reference=None):
    """Clear inherited tags, then use the common evaluator's R/S and E/Z labels.

    Quantization is the SAME float32 conversion as original_eligibility, so a
    near-degenerate stereo assignment cannot differ merely by diagnostic dtype.
    No bond-distance gate is consulted before assigning the 3D labels.
    """
    if isinstance(molecule, str):
        from .molecule import input_molecule
        molecule = input_molecule(molecule)
    if reference is None:
        reference = stereochemistry_signature(molecule)
    work = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(work)
    if hasattr(positions, 'detach'):
        positions = positions.detach().cpu().numpy()
    xyz = np.asarray(positions, dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError('nonfinite after common float32 quantization')
    observed = stereochemistry_signature(work, positions=xyz, from_geometry=True)
    atoms = dict(observed.chiral_centers)
    bonds = {(a, b): label for a, b, label in observed.double_bonds}
    atom_mismatch = [dict(atom_index=i, expected=label, observed=atoms.get(i))
                     for i, label in reference.chiral_centers if atoms.get(i) != label]
    bond_mismatch = [dict(bond_atoms=[a, b], expected=label, observed=bonds.get((a, b)))
                    for a, b, label in reference.double_bonds if bonds.get((a, b)) != label]
    rs = not atom_mismatch if reference.chiral_centers else None
    ez = not bond_mismatch if reference.double_bonds else None
    vacuous = not reference.chiral_centers and not reference.double_bonds
    return dict(stereo_evaluated=True, stereo_error=None,
        rs_retained=rs, ez_retained=ez,
        specified_stereo_retained=None if vacuous else not atom_mismatch and not bond_mismatch,
        rs_mismatches=atom_mismatch, ez_mismatches=bond_mismatch,
        observed_rs=[list(item) for item in observed.chiral_centers],
        observed_ez=[list(item) for item in observed.double_bonds])
