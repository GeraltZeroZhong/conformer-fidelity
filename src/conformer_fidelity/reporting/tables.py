"""Export supplied statistics and metric cells without recomputing observations."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _report(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def _write_new(output_dir, tables):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("export output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    for name, content in tables.items():
        (output / name).write_text(content, encoding="utf-8")


def statistics_tables(statistics):
    """Convert a completed report while preserving interval roles and full JSON."""
    report = _report(statistics)
    if (report.get("schema") != "CONFORMER_FIDELITY_PAIRED_METRIC_STATISTICS_V1"
            or report.get("status") != "complete"):
        raise ValueError("completed CONFORMER_FIDELITY_PAIRED_METRIC_STATISTICS_V1 report required")
    estimates, sensitivity, secondary = [], [], []
    for name in report["contrast_order"]:
        value = report["contrasts"][name]
        estimates.append(dict(contrast=name, mean=value["mean"], lower=value["ci"][0],
                              upper=value["ci"][1], alpha=value["alpha"],
                              confidence_level=value["confidence_level"], method=value["method"],
                              units=report["units"], scope=report["inference_scope"]))
        for role in ("paired_bootstrap_sensitivity", "cluster_sensitivity", "boundary_supplement"):
            value = report[role].get(name)
            interval = None if value is None else value.get("ci", value.get("interval"))
            sensitivity.append(dict(contrast=name, role=role,
                                    status="not_available_or_applicable" if value is None else "stored",
                                    lower=None if interval is None else interval[0],
                                    upper=None if interval is None else interval[1],
                                    replaces_contrast_bound=False, details_json=json_text(value).strip()))
    for method, metrics in report["secondary"].items():
        for metric, value in metrics.items():
            interval = value["ci"]
            secondary.append(dict(method=method, metric=metric, mean=value["mean"],
                                  lower=None if interval is None else interval[0],
                                  upper=None if interval is None else interval[1],
                                  nominal_alpha=value["nominal_alpha"], interval_method=value["method"],
                                  scope=report["secondary_scope"], details_json=json_text(value).strip()))
    return {
        "statistics.json": json_text(report),
        "contrasts.csv": csv_text(("contrast", "mean", "lower", "upper", "alpha",
                                    "confidence_level", "method", "units", "scope"), estimates),
        "sensitivities.csv": csv_text(("contrast", "role", "status", "lower", "upper",
                                        "replaces_contrast_bound", "details_json"), sensitivity),
        "metric-estimates.csv": csv_text(("method", "metric", "mean", "lower", "upper",
                                           "nominal_alpha", "interval_method", "scope", "details_json"), secondary),
    }


def export_statistics(statistics, output_dir):
    """Copy a supplied report to JSON/CSV without recomputing estimates or intervals."""
    tables = statistics_tables(statistics)
    index = dict(schema="CONFORMER_FIDELITY_STATISTICS_EXPORT_V1", status="complete",
                 files=list(tables), new_statistics_computed=False)
    tables["source-index.json"] = json_text(index)
    _write_new(output_dir, tables)
    return index


def export_metrics(input_path, output_dir, *, statistics=None, source_scope="caller-supplied metric observations"):
    """Export every NPZ cell with its original unit index and stratum.

    The table uses the identifiers and metric definitions supplied in the NPZ.
    Optional statistics are copied from a completed analysis report.
    """
    from conformer_fidelity.analysis import load_metrics

    if Path(output_dir).exists():
        raise FileExistsError("export output directory already exists; choose a new directory")
    cube = load_metrics(input_path)
    rows = [dict(index=i, stratum=str(cube.strata[i]), method=method, metric=metric,
                 value=float(cube.means[i, j, k]))
            for i in range(len(cube.means)) for j, method in enumerate(cube.names)
            for k, metric in enumerate(cube.fields)]
    tables = {"scaffold-metrics.csv": csv_text(("index", "stratum", "method", "metric", "value"), rows)}
    if statistics is not None:
        report = _report(statistics)
        if (report.get("units") != len(cube.means) or report.get("methods") != list(cube.names)
                or report.get("metrics") != list(cube.fields)):
            raise ValueError("supplied statistics do not align with the metric cube dimensions and labels")
        tables.update(statistics_tables(report))
    index = dict(schema="CONFORMER_FIDELITY_METRIC_EXPORT_V1", status="complete", scope=source_scope,
                 source=str(Path(input_path)), shape=list(cube.means.shape), methods=list(cube.names),
                 strata_source="input_array" if cube.strata_provided else "single_overall_group_default",
                 metrics=list(cube.fields), table_rows=len(rows), files=list(tables),
                 new_statistics_computed=False, new_distances_computed=False,
                 target_coordinate_files_read=False, original_values_preserved=True,
                 denominator_semantics="not supplied by this NPZ; no counts inferred",
                 identity_semantics="original unit index and stratum only; no molecule identity invented",
                 statistical_scope=None if statistics is None else report["inference_scope"],
                 statistical_data_identity_verified=False if statistics is not None else None)
    tables["source-index.json"] = json_text(index)
    _write_new(output_dir, tables)
    return index
