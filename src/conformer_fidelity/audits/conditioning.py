"""Compare upstream feature tensors for paired stereoisomer inputs.

Selected featurizer functions execute directly from the supplied local source.
This source-level audit runs independently of model weights and generation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from conformer_fidelity.chemistry import input_molecule

PAIRS = (
    ("enantiomers", "C[C@H](O)CC", "C[C@@H](O)CC"),
    ("diastereomers", "C[C@H](O)[C@H](F)CC", "C[C@H](O)[C@@H](F)CC"),
    ("double_bond", "CC/C=C/CC", "CC/C=C\\CC"),
)


def _upstream_package(source, name):
    source = Path(source).resolve()
    return source / name if (source / name).is_dir() else source


def _avgflow_functions(source):
    import scipy
    from scipy.linalg import sqrtm
    from scipy.sparse.linalg import eigs
    from rdkit.Chem.rdchem import ChiralType, BondType

    namespace = dict(np=np, scipy=scipy, sqrtm=sqrtm, eigs=eigs,
                     Chem=Chem, ChiralType=ChiralType, BT=BondType)
    provenance = {}
    for relative, names in (("dataloader/data_utils.py", None),
                            ("dataloader/preprocess.py", {"mol2features"})):
        path = source / relative
        chosen = []
        for node in ast.parse(path.read_text(), filename=str(path)).body:
            if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
                chosen.append(node)
                provenance[node.name] = dict(path=str(path), first_line=node.lineno,
                                             last_line=node.end_lineno)
            elif names is None and isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "chirality" for target in node.targets):
                chosen.append(node)
        if any(isinstance(node, ast.Name) and node.id in ("jax", "jnp")
               for chosen_node in chosen for node in ast.walk(chosen_node)):
            raise ValueError("This upstream featurizer needs JAX; the source-only audit supports NumPy functions")
        exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, provenance


def _etflow_functions(source):
    import torch
    from rdkit.Chem.rdchem import ChiralType

    path = source / "commons/utils.py"
    names = {"GetNumRings", "safe_index", "atom_to_feature_vector",
             "bond_to_feature_vector", "compute_edge_index", "get_chiral_tensors"}
    nodes = [node for node in ast.parse(path.read_text(), filename=str(path)).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id in {"chirality", "allowable_features"} for target in node.targets))]
    namespace = dict(torch=torch, np=np, Chem=Chem, ChiralType=ChiralType)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    provenance = {node.name: dict(path=str(path), first_line=node.lineno, last_line=node.end_lineno)
                  for node in nodes if isinstance(node, ast.FunctionDef)}
    return namespace, provenance


def _features(method, namespace, molecule, seed):
    if method == "etflow":
        edges, features = namespace["compute_edge_index"](molecule, with_edge_attr=True)
        return dict(z=np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()]),
                    node_attr=np.array([namespace["atom_to_feature_vector"](atom)
                                        for atom in molecule.GetAtoms()]),
                    edge_index=edges.numpy(), edge_attr=features.numpy())
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return namespace["mol2features"](molecule, "drugs")
    finally:
        np.random.set_state(state)


def _indexed_graph(molecule):
    return ([tuple((a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                    a.GetTotalNumHs(includeNeighbors=True))) for a in molecule.GetAtoms()],
            sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                   for b in molecule.GetBonds()))


def _compare(first, second):
    if first.keys() != second.keys():
        raise ValueError("The upstream featurizer returned different tensor fields")
    return {name: dict(equal=bool(np.array_equal(first[name], second[name])),
                       shape_first=list(first[name].shape), shape_second=list(second[name].shape),
                       maximum_abs_difference=float(np.max(np.abs(first[name] - second[name])))
                       if first[name].shape == second[name].shape and first[name].size else None)
            for name in first}


def audit_conditioning(method, source, *, pairs=PAIRS, seed=0):
    """Measure feature equality for explicit same-index stereoisomer pairs.

    A shared NumPy seed couples positional-encoding randomness across each
    pair. The report records tensor equality, shapes and maximum differences
    for the selected source functions and inputs.
    """
    if method not in {"avgflow", "etflow"}:
        raise ValueError("Conditioning source audit supports avgflow or etflow")
    source = _upstream_package(source, method)
    namespace, provenance = (_avgflow_functions(source) if method == "avgflow"
                             else _etflow_functions(source))
    rows = []
    for name, first, second in pairs:
        a, b = input_molecule(first), input_molecule(second)
        if _indexed_graph(a) != _indexed_graph(b):
            raise ValueError("A conditioning pair must preserve indexed non-stereo chemistry")
        canonical = [Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True) for mol in (a, b)]
        if canonical[0] == canonical[1]:
            raise ValueError("A conditioning pair must specify distinct stereoisomers")
        tensors = _compare(_features(method, namespace, a, seed),
                           _features(method, namespace, b, seed))
        rows.append(dict(name=name, smiles=[first, second], tensors=tensors,
                         all_tensors_identical=all(item["equal"] for item in tensors.values())))
    control = _compare(*[_features(method, namespace, input_molecule(smiles), seed)
                         for smiles in ("CCCO", "CCCN")])
    return dict(method=method, source=str(source), source_functions=provenance,
                status="measured", seed=seed, pairs=rows, chemical_positive_control=control,
                chemical_positive_control_changes=not all(v["equal"] for v in control.values()),
                rdkit_version=rdBase.rdkitVersion, numpy_version=np.__version__,
                model_calls=0, conformers_read_or_generated=0,
                scope="Tensor comparison for the specified source functions and paired inputs")
