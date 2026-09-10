import json

import numpy as np
import pytest

from conformer_fidelity.artifacts import load_ensemble, save_ensemble, save_json


def test_ensemble_roundtrip_keeps_failed_slots(tmp_path):
    xyz = np.full((2, 3, 3), np.nan)
    xyz[0] = 0
    path = tmp_path / "sample.npz"
    save_ensemble(path, "O", dict(coordinates=xyz, present_mask=[True, False], status=[0, None]))
    restored = load_ensemble(path)
    np.testing.assert_array_equal(restored["coordinates"], xyz)
    assert restored["present_mask"].tolist() == [True, False]
    assert restored["metadata"]["status"] == [0, None]
    with pytest.raises(FileExistsError):
        save_ensemble(path, "O", dict(coordinates=xyz, present_mask=[True, False]))


def test_absent_coordinates_are_not_fabricated(tmp_path):
    with pytest.raises(ValueError, match="Absent"):
        save_ensemble(tmp_path / "bad.npz", "O", dict(coordinates=np.zeros((1, 3, 3)), present_mask=[False]))


def test_documented_copy_preserves_raw_arrays_and_metadata(tmp_path):
    result = dict(coordinates=np.ones((1, 3, 3)), present_mask=[True],
                  raw_coordinates=np.zeros((1, 3, 3)), raw_present_mask=[True], native_status=0)
    original, copied = tmp_path / "original.npz", tmp_path / "copied.npz"
    save_ensemble(original, "O", result)
    loaded = load_ensemble(original)
    save_ensemble(copied, loaded["smiles"], loaded)
    restored = load_ensemble(copied)
    np.testing.assert_array_equal(restored["raw_coordinates"], result["raw_coordinates"])
    assert restored["raw_present_mask"].tolist() == [True]
    assert restored["metadata"] == {"native_status": 0}


def test_json_arrays_missing_values_and_no_clobber(tmp_path):
    path = tmp_path / "result.json"
    save_json(path, {"array": np.array([0.25, np.nan]), "n": np.int64(2)})
    assert json.loads(path.read_text()) == {"array": [0.25, None], "n": 2}
    with pytest.raises(FileExistsError):
        save_json(path, {})
