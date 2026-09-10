"""Target-blind fixed-pool representative selection, not target-coverage optimization.

Six deterministic rules select from an unchanged candidate pool using only
original eligibility, pairwise heavy-atom RMSD and single-point energy.
Radius rules may return fewer candidates than requested and do not refill.
"""
from .geometry import pool_geometry, select_conformers
from .rules import METHODS, select_indices

__all__ = ['METHODS', 'pool_geometry', 'select_conformers', 'select_indices']
