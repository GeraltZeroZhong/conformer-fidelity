"""Portable NumPy coordinate arrays with JSON metadata."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np


def json_value(value):
    """Convert scientific scalar/array results into ordinary JSON values."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Result contains an unsupported JSON value: {type(value).__name__}")


def save_json(path, value):
    """Write JSON to a new output path."""
    path = Path(path)
    text = json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def save_ensemble(path, smiles, result):
    """Store coordinates in the atom order of ``Chem.AddHs(MolFromSmiles(smiles))``.

    ``present_mask`` records returned slots, not geometric validity. Missing
    slots remain NaN. A result may also preserve native raw coordinates.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Ensemble output must use the .npz extension")
    coordinates = np.asarray(result["coordinates"], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [slots, atoms, 3]")
    present = np.asarray(result.get("present_mask", np.ones(len(coordinates), bool)), dtype=bool)
    if present.shape != (len(coordinates),):
        raise ValueError("present_mask must have one value per slot")
    if not np.isnan(coordinates[~present]).all():
        raise ValueError("Absent slots must contain NaN coordinates")
    arrays = dict(coordinates=coordinates, present_mask=present, smiles=np.asarray(smiles))
    for name in ("raw_coordinates", "raw_present_mask"):
        if name in result:
            arrays[name] = np.asarray(result[name])
    metadata = dict(result.get("metadata", {}))
    metadata.update({key: value for key, value in result.items()
                     if key not in {"molecule", "smiles", "metadata", "coordinates", "present_mask",
                                    "raw_coordinates", "raw_present_mask"}})
    arrays["metadata_json"] = np.asarray(json.dumps(json_value(metadata), allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_ensemble(path):
    """Read the public ensemble format without requiring a local checkpoint."""
    with np.load(path, allow_pickle=False) as data:
        coordinates = np.array(data["coordinates"], dtype=np.float64)
        smiles = str(data["smiles"].item())
        present = np.array(data["present_mask"], dtype=bool)
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        raw = {name: np.array(data[name]) for name in ("raw_coordinates", "raw_present_mask") if name in data}
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [slots, atoms, 3]")
    if present.shape != (len(coordinates),):
        raise ValueError("present_mask must have one value per slot")
    if not np.isnan(coordinates[~present]).all():
        raise ValueError("Absent slots must contain NaN coordinates")
    return dict(smiles=smiles, coordinates=coordinates, present_mask=present, metadata=metadata, **raw)
