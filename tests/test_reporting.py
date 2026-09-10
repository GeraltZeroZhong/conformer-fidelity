import csv
import json

import numpy as np
import pytest

from conformer_fidelity.analysis import Contrast, analyze_metrics
from conformer_fidelity.reporting import export_metrics, export_statistics, statistics_tables


def fixture(tmp_path):
    means = np.arange(12, dtype=float).reshape(4, 3, 1)/12
    names, fields, strata = ["a", "b", "c"], ["rate"], ["high"]*2+["low"]*2
    path = tmp_path / "metrics.npz"
    np.savez(path, means=means, names=names, fields=fields, strata=strata)
    stats = analyze_metrics(means, names, fields, strata,
                            contrasts=[Contrast("delta", "b", "rate", "a", "rate", (-1, 1))],
                            metric_domains={"rate": (0, 1)}, draws=8)
    return path, means, stats


def test_export_preserves_every_original_value_and_order(tmp_path):
    path, means, _ = fixture(tmp_path)
    report = export_metrics(path, tmp_path / "tables")
    rows = list(csv.DictReader((tmp_path / "tables/scaffold-metrics.csv").open()))
    assert len(rows) == means.size == report["table_rows"]
    assert [float(row["value"]) for row in rows] == means.ravel().tolist()
    assert [int(row["index"]) for row in rows] == [0]*3+[1]*3+[2]*3+[3]*3
    assert report["new_statistics_computed"] is False
    assert report["statistical_scope"] is None
    assert report["denominator_semantics"].startswith("not supplied")
    assert not (tmp_path / "tables/contrasts.csv").exists()


def test_optional_statistics_preserves_full_json_and_separate_roles(tmp_path):
    path, _, stats = fixture(tmp_path)
    saved = tmp_path / "input-statistics.json"
    saved.write_text(json.dumps(stats))
    report = export_metrics(path, tmp_path / "tables", statistics=saved)
    assert json.loads((tmp_path / "tables/statistics.json").read_text()) == stats
    sensitivity = list(csv.DictReader((tmp_path / "tables/sensitivities.csv").open()))
    assert {row["role"] for row in sensitivity} == {
        "paired_bootstrap_sensitivity", "cluster_sensitivity", "boundary_supplement"}
    assert all(row["replaces_contrast_bound"] == "False" for row in sensitivity)
    assert report["statistical_data_identity_verified"] is False
    assert report["statistical_scope"] == "caller_defined_bounded_contrasts"


def test_descriptive_only_export_does_not_invent_formal_rows(tmp_path):
    stats = analyze_metrics(np.zeros((4, 1, 1)), ["a"], ["rmsd"], ["x"]*4, draws=4)
    export_statistics(stats, tmp_path / "out")
    assert list(csv.DictReader((tmp_path / "out/contrasts.csv").open())) == []
    row = list(csv.DictReader((tmp_path / "out/metric-estimates.csv").open()))[0]
    assert row["lower"] == row["upper"] == ""


def test_existing_output_rejected_without_changes(tmp_path):
    path, _, stats = fixture(tmp_path)
    output = tmp_path / "out"
    export_metrics(path, output)
    previous = (output / "source-index.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_metrics(path, output)
    with pytest.raises(FileExistsError):
        export_statistics(stats, output)
    assert (output / "source-index.json").read_bytes() == previous


def test_unsupported_report_schema_rejected(tmp_path):
    with pytest.raises(ValueError, match="CONFORMER_FIDELITY_PAIRED_METRIC_STATISTICS_V1"):
        statistics_tables({"schema": "UNSUPPORTED_SCHEMA", "status": "complete"})


def test_misaligned_statistics_rejected_before_writing(tmp_path):
    path, _, stats = fixture(tmp_path)
    stats["methods"] = ["other"]
    with pytest.raises(ValueError, match="align"):
        export_metrics(path, tmp_path / "out", statistics=stats)
    assert not (tmp_path / "out").exists()


def test_missing_strata_export_is_explicit(tmp_path):
    path = tmp_path / "metrics.npz"
    np.savez(path, means=np.zeros((2, 1, 1)), names=["a"], fields=["x"])
    index = export_metrics(path, tmp_path / "out")
    assert index["strata_source"] == "single_overall_group_default"
    rows = list(csv.DictReader((tmp_path / "out/scaffold-metrics.csv").open()))
    assert {row["stratum"] for row in rows} == {"overall"}
