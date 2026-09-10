"""Statistical contracts exercised with synthetic arrays."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import t

from conformer_fidelity.analysis import (
    Contrast, MetricCube, analyze_metrics, bootstrap_means, boundary_interval,
    cluster_sandwich, hoeffding_interval, load_metrics, repeat_transition_counts,
    run_statistics, stratum_weights,
)


def fixture():
    means = np.zeros((8, 2, 3))
    means[:, 0, 0] = [.2, .4, .6, .8, .3, .5, .7, .9]
    means[:, 0, 1] = means[:, 0, 0] - .1
    means[:, 1, 1] = means[:, 0, 1] + .05
    means[:, :, 2] = 100  # Missing-output RMSD sentinel, not a known range.
    return means, ("native", "post"), ("finite", "qualified", "rmsd"), np.array(["high"]*4+["low"]*4)


def contrasts():
    return [Contrast("gap", "native", "finite", "native", "qualified", (0, 1)),
            Contrast("change", "post", "qualified", "native", "qualified", (-1, 1))]


def test_fixed_hoeffding_width_uses_independent_unit_denominator():
    units = 32
    values, weights = np.zeros(units), np.full(units, 1/units)
    for domain in ((0, 1), (-1, 1)):
        value = hoeffding_interval(values, weights, domain, .0125)
        width = (domain[1]-domain[0]) * np.sqrt(np.log(160)/(2*units))
        assert value["unclipped_half_width"] == pytest.approx(width)
        assert value["ci"] == pytest.approx([max(domain[0], -width), min(domain[1], width)])


def test_unequal_samples_keep_equal_stratum_mass():
    weights = stratum_weights(["high"]*2+["low"]*6)
    assert weights[:2].sum() == pytest.approx(.5)
    assert weights[2:].sum() == pytest.approx(.5)
    weighted = stratum_weights(["a"]*2+["b"]*6, {"a": .2, "b": .8})
    assert weighted[:2].sum() == pytest.approx(.2)


def test_paired_bootstrap_preserves_correlated_columns_and_pcg64_draw_order():
    data = np.arange(8.0)[:, None]
    data = np.column_stack((data, data+10))
    strata = ["high"]*4+["low"]*4
    result = bootstrap_means(data, strata, 42, 5)
    rng = np.random.Generator(np.random.PCG64(42))
    high = data[:4][rng.integers(4, size=(5, 4))].mean(1)
    low = data[4:][rng.integers(4, size=(5, 4))].mean(1)
    np.testing.assert_array_equal(result["high"], high)
    np.testing.assert_array_equal(result["low"], low)
    np.testing.assert_array_equal(result["overall"], (high+low)/2)
    np.testing.assert_array_equal(result["overall"][:, 1]-result["overall"][:, 0], np.full(5, 10))


def test_explicit_contrasts_sign_domains_and_separate_sensitivities():
    report = analyze_metrics(*fixture(), contrasts=contrasts(), draws=12, seed=7,
                             cluster_labels=np.arange(8), metric_domains={"qualified": (0, 1)})
    assert report["contrasts"]["gap"]["mean"] == pytest.approx(.1)
    assert report["contrasts"]["change"]["mean"] == pytest.approx(.05)
    assert report["contrasts"]["gap"]["alpha"] == .025
    assert report["contrasts"]["gap"]["method"] == "fixed_bounded_hoeffding"
    assert report["secondary"]["native"]["rmsd"]["ci"] is None
    assert report["secondary"]["native"]["rmsd"]["boundary_supplement"] is None
    assert report["experiment_preregistration_verified"] is False
    assert report["independent_units_verified"] is False
    assert set(report["paired_bootstrap_sensitivity"]) == {"gap", "change"}
    assert set(report["cluster_sensitivity"]) == {"gap", "change"}


def test_no_contrast_is_descriptive():
    report = analyze_metrics(*fixture(), draws=5)
    assert report["inference_scope"] == "descriptive_only"
    assert report["contrasts"] == {}
    assert report["conditional_familywise_coverage_lower_bound"] is None
    assert report["bootstrap_seed"] == 0


def test_boundary_supplement_nonzero_and_primary_not_replaced():
    weights = np.full(8, 1/8)
    primary = hoeffding_interval(np.zeros(8), weights, (-1, 1), .0125)
    boundary = boundary_interval(np.zeros(8), weights, (-1, 1), .0125)
    assert primary["ci"][1] > 0
    assert boundary["interval"][0] < 0 < boundary["interval"][1]
    assert "not_necessarily_no_repeat_transitions" in boundary["event"]
    assert boundary_interval(np.arange(8)/8, weights, (0, 1), .05) is None


def test_cr1_matches_two_stratum_formula_and_degenerate_scope():
    values = np.array([0., .5, .3, .7, .2, .8, .1, .4])
    strata = np.array(["high"]*4+["low"]*4)
    groups = np.array([0, 0, 1, 2, 2, 3, 3, 4])
    result = cluster_sandwich(values, strata, groups, .0125, (0, 1))
    residual = values.copy()
    residual[:4] -= values[:4].mean()
    residual[4:] -= values[4:].mean()
    scores = np.array([np.sum(residual[groups == i]/8) for i in range(5)])
    correction = 5/4*7/6
    se = np.sqrt(correction*(scores @ scores))
    half = t.ppf(1-.0125/2, 4)*se
    assert result["standard_error"] == pytest.approx(se)
    assert result["ci"] == pytest.approx([max(0, values.mean()-half), min(1, values.mean()+half)])
    flat = cluster_sandwich(np.full(8, .5), strata, np.arange(8), domain=(0, 1))
    assert flat["ci"] == [0, 1]
    assert "zero sandwich variance" in flat["unavailable"]


def test_repeat_cancellation_not_no_events_and_boolean_arrays_supported():
    native = np.array([[0, 1], [1, 0]], dtype=bool)
    post = ~native
    report = repeat_transition_counts(native, post)
    assert report["gain"] == report["loss"] == 2
    assert report["zero_mean_with_offsetting_repeat_transitions"] == 2
    assert report["all_unit_mean_changes_zero"] is True
    assert report["no_repeat_transitions"] is False


def test_missing_component_label_is_not_silently_excluded():
    with pytest.raises(ValueError, match="nonmissing"):
        cluster_sandwich([0, .2, .4, .8], ["x"]*4, [0, 1, np.nan, 2], domain=(0, 1))


def test_arbitrary_cube_shape_loads_and_analyzes(tmp_path):
    path = tmp_path / "scaffold-metrics.npz"
    np.savez(path, means=np.zeros((16, 3, 4)), names=[f"m{i}" for i in range(3)],
             fields=[f"metric{i}" for i in range(4)], strata=["high"]*8+["low"]*8)
    cube = load_metrics(path)
    assert cube.means.shape == (16, 3, 4)
    report = run_statistics(path, tmp_path / "analysis", draws=4)
    assert report["units"] == 16
    assert report["input"]["coordinates_read"] is False
    assert json.loads((tmp_path / "analysis/statistics.json").read_text())["status"] == "complete"


def test_npz_without_strata_records_single_group_default(tmp_path):
    path = tmp_path / "metrics.npz"
    np.savez(path, means=np.zeros((4, 1, 1)), names=["a"], fields=["rate"])
    report = run_statistics(path, tmp_path / "out", draws=4)
    assert report["stratum_counts"] == {"overall": 4}
    assert report["input"]["strata_source"] == "single_overall_group_default"
    with pytest.raises(FileExistsError):
        run_statistics(path, tmp_path / "out", draws=4)


@pytest.mark.parametrize("change", ["nan", "duplicate", "shape"])
def test_bad_cube_fails_without_silent_dropping(change):
    means, names, fields, strata = fixture()
    if change == "nan":
        means[0, 0, 0] = np.nan
    elif change == "duplicate":
        names = ("native", "native")
    else:
        strata = strata[:-1]
    with pytest.raises(ValueError, match="aligned"):
        MetricCube(means, names, fields, strata)


def test_unknown_and_out_of_bound_contrasts_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        analyze_metrics(*fixture(), contrasts=[Contrast("bad", "missing", "finite", "post", "finite", (-1, 1))], draws=3)
    with pytest.raises(ValueError, match="bounded"):
        analyze_metrics(*fixture(), contrasts=[Contrast("bad", "native", "finite", "native", "qualified", (-1, 0))], draws=3)
    with pytest.raises(ValueError, match="unknown fields"):
        analyze_metrics(*fixture(), metric_domains={"other": (0, 1)}, draws=3)


def test_import_from_outside_checkout_needs_only_public_source(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    command = [sys.executable, "-c", "import sys; from conformer_fidelity.analysis import run_statistics; "
               "assert not any(n in sys.modules for n in ('torch', 'jax', 'posebusters'))"]
    import os
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
