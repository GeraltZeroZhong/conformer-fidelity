import numpy as np
import pytest

from conformer_fidelity.audits.conditioning import _compare, audit_conditioning


def test_tensor_measurement_does_not_require_a_collision():
    assert _compare({"node": np.array([1])}, {"node": np.array([2])})["node"]["equal"] is False


def test_avgflow_original_functions_without_jax_or_weights(tmp_path):
    source = tmp_path / "avgflow/dataloader"
    source.mkdir(parents=True)
    (source / "data_utils.py").write_text("def atoms(mol):\n    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()])\n")
    (source / "preprocess.py").write_text("def mol2features(mol, dataset):\n    return {'node': atoms(mol)}\n")
    report = audit_conditioning("avgflow", source.parent, seed=3)
    assert report["chemical_positive_control_changes"]
    assert len(report["pairs"]) == 3
    assert all(row["all_tensors_identical"] for row in report["pairs"])
    assert report["model_calls"] == 0


def test_missing_upstream_source_is_not_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_conditioning("avgflow", tmp_path)
