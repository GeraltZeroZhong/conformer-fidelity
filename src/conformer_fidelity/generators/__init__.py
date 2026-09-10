"""Target-blind generators; optional neural/CDPKit dependencies load on use.

All outputs use the explicit-H atom order of the input SMILES. Missing requests
are NaN-padded with a false present_mask; returned nonfinite slots remain present.
"""

from .etkdg import generate, relax

__all__ = ['generate', 'relax']
