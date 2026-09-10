import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from conformer_fidelity.cli import main


def test_help_does_not_import_backends(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c",
        "import sys; from conformer_fidelity.cli import parser; app = parser(); "
        "assert app.prog == 'conformer-fidelity'; app.format_help(); "
        "assert not any(x in sys.modules for x in ('torch','jax','posebusters','CDPL'))"],
        env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_neural_assets_does_not_start_download(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--method", "etflow", "--smiles", "CCO", "--output", str(tmp_path / "x.npz")])
    assert exc.value.code == 2
    assert "--source and --checkpoint" in capsys.readouterr().err


def test_cpu_demo(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output)]) == 0
    assert json.loads((output / "example.json").read_text())["benchmark"] is False
    assert len(json.loads((output / "identity.json").read_text())["slots"]) == 20
    assert (output / "scores.json").is_file()
    with pytest.raises(SystemExit):
        main(["demo", "--output", str(output)])


def test_select_cli_keeps_requested_slots_without_refill(tmp_path):
    from conformer_fidelity.artifacts import load_ensemble, save_ensemble
    from conformer_fidelity.generators.etkdg import generate

    generated = generate("CCO", count=1, seed=7)
    source, output = tmp_path / "source.npz", tmp_path / "selected.npz"
    save_ensemble(source, "CCO", generated)
    assert main(["select", "--input", str(source), "--output", str(output),
                 "--method", "energy", "--count", "3"]) == 0
    selected = load_ensemble(output)
    assert selected["coordinates"].shape[0] == 3
    assert selected["present_mask"].tolist() == [True, False, False]
    assert selected["metadata"]["selection"]["indices"] == [0]
    assert selected["metadata"]["target_access"] is False
    np.testing.assert_array_equal(selected["coordinates"][0], generated["coordinates"][0])
    assert np.isnan(selected["coordinates"][1:]).all()
