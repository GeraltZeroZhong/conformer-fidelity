"""Paired, stratified analysis of scaffold-level mean metrics.

Input arrays have shape [units, methods, metrics]. Hoeffding intervals use
independent bounded units, fixed stratum weights and prespecified contrasts.
Paired bootstrap and cluster-robust intervals provide sensitivity analyses.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class Contrast:
    """A paired left-minus-right contrast with an externally known value range."""

    name: str
    left_method: str
    left_metric: str
    right_method: str
    right_metric: str
    domain: tuple[float, float]


@dataclass(frozen=True)
class MetricCube:
    """Aligned means, names, fields and strata; values include failed-unit sentinels."""

    means: np.ndarray
    names: tuple[str, ...]
    fields: tuple[str, ...]
    strata: np.ndarray
    strata_provided: bool = True

    def __post_init__(self):
        means = np.asarray(self.means, dtype=np.float64)
        strata = np.asarray(self.strata, dtype=str)
        names, fields = tuple(self.names), tuple(self.fields)
        if (
            means.ndim != 3 or means.shape[1:] != (len(names), len(fields))
            or not len(means) or not names or not fields
            or strata.shape != (len(means),) or not np.isfinite(means).all()
            or len(set(names)) != len(names) or len(set(fields)) != len(fields)
            or not all(isinstance(s, str) and s for s in names + fields)
            or not np.all(strata != "")
            or ("overall" in strata and len(set(strata.tolist())) > 1)
        ):
            raise ValueError("finite aligned unit/method/metric cube and unique labels required")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "fields", fields)


def load_metrics(path: str | Path) -> MetricCube:
    """Read means/names/fields and optional strata from an explicit NPZ file.

    Missing strata produce one explicitly recorded ``overall`` group.
    """
    with np.load(path, allow_pickle=False) as saved:
        strata_provided = "strata" in saved.files
        strata = saved["strata"] if strata_provided else np.repeat("overall", len(saved["means"]))
        return MetricCube(saved["means"], tuple(saved["names"].tolist()),
                          tuple(saved["fields"].tolist()), strata, strata_provided)


def _proportions(strata, proportions=None):
    strata = np.asarray(strata, dtype=str)
    if strata.ndim != 1 or not len(strata) or not np.all(strata != ""):
        raise ValueError("nonempty one-dimensional stratum labels required")
    groups = sorted(set(strata.tolist()))
    if "overall" in groups and len(groups) > 1:
        raise ValueError("overall is allowed only for an unstratified, single-group input")
    masses = dict.fromkeys(groups, 1.0 / len(groups)) if proportions is None else dict(proportions)
    if (set(masses) != set(groups) or not np.isfinite(list(masses.values())).all()
            or any(v < 0 for v in masses.values())
            or not np.isclose(sum(masses.values()), 1, rtol=0, atol=1e-12)):
        raise ValueError("fixed nonnegative proportions must cover all strata and sum to one")
    return strata, groups, masses


def stratum_weights(strata, proportions=None):
    """Assign equal total weight to each stratum by default."""
    strata, groups, masses = _proportions(strata, proportions)
    counts = {group: int(np.sum(strata == group)) for group in groups}
    if min(counts.values()) < 2:
        raise ValueError("at least two independent units per stratum required")
    return np.array([masses[group] / counts[group] for group in strata])


def bootstrap_means(data, strata, seed, draws=50000, *, proportions=None):
    """Paired within-stratum bootstrap using seeded PCG64 sampling.

    All methods and metrics share the same sampled unit indices. Percentile
    intervals from these draws provide an approximate sensitivity analysis.
    """
    data = np.asarray(data, dtype=np.float64)
    strata, groups, masses = _proportions(strata, proportions)
    if (not np.isfinite(data).all() or data.ndim < 1 or len(data) != len(strata)
            or type(draws) is not int or draws < 2):
        raise ValueError("complete paired data and at least two bootstrap draws required")
    stratum_weights(strata, masses)
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = {}
    for group in groups:
        values = data[strata == group]
        n = len(values)
        output = np.empty((draws,) + values.shape[1:])
        for start in range(0, draws, 100):
            b = min(100, draws - start)
            ix = rng.integers(n, size=(b, n))
            counts = np.zeros((b, n))
            np.add.at(counts, (np.arange(b)[:, None], ix), 1.0)
            output[start:start+b] = (counts @ values.reshape(n, -1) / n).reshape(
                (b,) + values.shape[1:])
        boot[group] = output
    boot["overall"] = sum(masses[group] * boot[group] for group in groups)
    return boot


def _bounded(values, weights, domain, alpha):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    low, high = map(float, domain)
    if (values.ndim != 1 or not len(values) or values.shape != weights.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or not np.isfinite([low, high]).all() or not low < high
            or np.any((values < low) | (values > high)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
            or not 0 < alpha < 1):
        raise ValueError("finite bounded values and fixed normalized nonnegative weights required")
    return values, weights, low, high


def hoeffding_interval(values, weights, domain, alpha):
    """Fixed weighted two-sided Hoeffding bound for independent bounded units."""
    values, weights, low, high = _bounded(values, weights, domain, alpha)
    weight_square_sum = float(weights @ weights)
    halfwidth = float((high-low) * np.sqrt(np.log(2/alpha) * weight_square_sum / 2))
    mean = float(weights @ values)
    return dict(ci=[max(low, mean-halfwidth), min(high, mean+halfwidth)],
                method="fixed_bounded_hoeffding", unclipped_half_width=halfwidth,
                bounded_domain=[low, high], weight_square_sum=weight_square_sum)


def boundary_interval(values, weights, domain, alpha, *, maximum_group_weight=None):
    """Calculate a separate interval supplement for zero or boundary events."""
    values, weights, low, high = _bounded(values, weights, domain, alpha)
    if not (np.all(values == 0) or np.all(values == low) or np.all(values == high)):
        return None
    if maximum_group_weight is not None:
        if not 0 < maximum_group_weight <= 1:
            raise ValueError("maximum group weight must be in (0, 1]")
        upper = min(1.0, float(maximum_group_weight) * np.log(1/alpha))
        method = "independent_component_max_weight_zero_event_bound"
    elif np.allclose(weights, 1/len(values), rtol=0, atol=1e-15):
        upper = float(-np.expm1(np.log(alpha)/len(values)))
        method = "independent_scaffold_balanced_zero_event_bound"
    else:
        upper = min(1.0, float(weights.max()) * np.log(1/alpha))
        method = "independent_scaffold_max_weight_zero_event_bound"
    if low < 0 < high and np.all(values == 0):
        interval = [low*upper, high*upper]
        event = "all_scaffold_mean_changes_zero_not_necessarily_no_repeat_transitions"
    elif np.all(values == low):
        interval, event = [low, low+(high-low)*upper], "all_at_lower_boundary"
    else:
        interval, event = [high-(high-low)*upper, high], "all_at_upper_boundary"
    return dict(interval=[float(v) for v in interval], method=method,
                event=event, alpha=float(alpha), event_probability_upper=float(upper))


def cluster_sandwich(values, strata, labels, alpha=0.05, domain=(-1.0, 1.0), *, proportions=None):
    """Stratum-intercept CR1/t sensitivity with caller-supplied cluster labels."""
    strata, strata_names, masses = _proportions(strata, proportions)
    weights = stratum_weights(strata, masses)
    values, weights, low, high = _bounded(values, weights, domain, alpha)
    labels = np.asarray(labels)
    if (labels.shape != values.shape
            or (np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all())
            or (np.issubdtype(labels.dtype, np.str_) and np.any(labels == ""))):
        raise ValueError("aligned nonmissing component labels required")
    groups = np.unique(labels)
    group_weights = np.array([weights[labels == group].sum() for group in groups])
    mean = float(weights @ values)
    report = dict(mean=mean, groups=len(groups), degrees_of_freedom=len(groups)-1,
                  maximum_group_weight=float(group_weights.max()),
                  scope="approximate independent-component sensitivity with fixed stratum weights")
    boundary = boundary_interval(values, weights, domain, alpha,
                                 maximum_group_weight=group_weights.max())
    if boundary is not None:
        return dict(report, ci=boundary["interval"], boundary=boundary, standard_error=None)
    if len(groups) < 2:
        return dict(report, ci=[low, high], standard_error=None, unavailable="only one component")
    residual = values.copy()
    for group in strata_names:
        residual[strata == group] -= values[strata == group].mean()
    scores = np.array([np.sum(weights[labels == group] * residual[labels == group])
                       for group in groups])
    correction = len(groups)/(len(groups)-1) * (len(values)-1)/(len(values)-len(strata_names))
    se = float(np.sqrt(correction * (scores @ scores)))
    halfwidth = float(t.ppf(1-alpha/2, len(groups)-1) * se)
    interval = [max(low, mean-halfwidth), min(high, mean+halfwidth)]
    if se == 0:
        interval = [low, high]
        report["unavailable"] = "zero sandwich variance outside fixed boundary cases"
    return dict(report, ci=interval, standard_error=se, cr1_correction=float(correction))


def _bootstrap_interval(bootstrap, domain, alpha):
    interval = np.quantile(bootstrap, [alpha/2, 1-alpha/2]).tolist()
    if interval[0] == interval[1]:
        return dict(ci=list(domain) if domain is not None else None,
                    method="uninformative_degenerate_bootstrap",
                    note="empirical zero variance does not establish population equality")
    return dict(ci=interval, method="paired_stratified_percentile_bootstrap")


def _contrast_values(cube, contrasts):
    if len({contrast.name for contrast in contrasts}) != len(contrasts):
        raise ValueError("unique contrast names required")
    values = []
    for contrast in contrasts:
        if not contrast.name:
            raise ValueError("nonempty contrast name required")
        try:
            left = cube.means[:, cube.names.index(contrast.left_method),
                              cube.fields.index(contrast.left_metric)]
            right = cube.means[:, cube.names.index(contrast.right_method),
                               cube.fields.index(contrast.right_metric)]
        except ValueError as exc:
            raise ValueError(f"unknown method or metric in contrast {contrast.name}") from exc
        values.append(left-right)
    return np.column_stack(values) if values else np.empty((len(cube.means), 0))


def analyze_metrics(
    means, names, fields, strata, *, contrasts: Sequence[Contrast | Mapping] = (),
    metric_domains=None, stratum_proportions=None, cluster_labels=None,
    draws=50000, seed=0, alpha=0.05,
):
    """Analyze paired metrics and optional prespecified bounded contrasts.

    Empty contrasts give descriptive statistics. Supply theoretical metric
    domains separately from any failure sentinels in the observations.
    K contrasts use alpha/K Bonferroni-adjusted Hoeffding bounds.
    """
    cube = MetricCube(means, tuple(names), tuple(fields), strata)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    contrasts = tuple(c if isinstance(c, Contrast) else Contrast(**c) for c in contrasts)
    domains = {} if metric_domains is None else dict(metric_domains)
    if set(domains) - set(cube.fields):
        raise ValueError("metric domains contain unknown fields")
    _, groups, masses = _proportions(cube.strata, stratum_proportions)
    weights = stratum_weights(cube.strata, masses)
    values = _contrast_values(cube, contrasts)
    # Validate observations against the caller's theoretical bounds.
    for index, contrast in enumerate(contrasts):
        _bounded(values[:, index], weights, contrast.domain, alpha)
    for field, domain in domains.items():
        if domain is not None:
            for j in range(len(cube.names)):
                _bounded(cube.means[:, j, cube.fields.index(field)], weights, domain, alpha)
    boot = bootstrap_means(cube.means, cube.strata, seed, draws, proportions=masses)
    boot_cube = MetricCube(boot["overall"], cube.names, cube.fields, np.repeat("draw", draws))
    boot_contrasts = _contrast_values(boot_cube, contrasts)
    report = dict(
        schema="CONFORMER_FIDELITY_PAIRED_METRIC_STATISTICS_V1", status="complete",
        units=len(cube.means), unit_axis="already averaged scaffold/unit rows",
        methods=list(cube.names), metrics=list(cube.fields),
        stratum_counts=dict(Counter(cube.strata.tolist())), stratum_proportions=masses,
        bootstrap_seed=seed, bootstrap_draws=draws, alpha=alpha,
        contrast_order=[c.name for c in contrasts], contrasts={}, secondary={},
        paired_bootstrap_sensitivity={}, boundary_supplement={}, cluster_sensitivity={},
        contrast_definitions=[dict(asdict(c), domain=list(c.domain)) for c in contrasts],
        metric_domains={key: None if value is None else list(value) for key, value in domains.items()},
        inference_scope="descriptive_only" if not contrasts else "caller_defined_bounded_contrasts",
        experiment_preregistration_verified=False, independent_units_verified=False,
        conditional_familywise_coverage_lower_bound=None if not contrasts else 1-alpha,
        coverage_assumptions=["independent bounded unit outcomes, not necessarily identically distributed",
            "fixed stratum weights, externally known bounds, fixed contrasts and fixed sample size",
            "no optional stopping or outcome-dependent contrast selection"],
        bootstrap_scope="approximate paired stratified percentile sensitivity; sparse outcomes can undercover",
        secondary_scope="pointwise nominal descriptive bootstrap intervals; no simultaneous winner claim",
        sensitivity_combination="no sensitivity or boundary supplement replaces or narrows contrast bounds",
        failed_units_dropped=False, repeats_resampled_as_independent_units=False,
    )
    if cluster_labels is not None and np.asarray(cluster_labels).shape != (len(cube.means),):
        raise ValueError("component labels must align with unit rows")
    for j, contrast in enumerate(contrasts):
        local_alpha = alpha / len(contrasts)
        value = values[:, j]
        report["contrasts"][contrast.name] = dict(
            mean=float(weights @ value), alpha=local_alpha, confidence_level=1-local_alpha,
            **hoeffding_interval(value, weights, contrast.domain, local_alpha))
        report["paired_bootstrap_sensitivity"][contrast.name] = dict(
            mean=float(weights @ value), nominal_alpha=local_alpha,
            **_bootstrap_interval(boot_contrasts[:, j], contrast.domain, local_alpha))
        report["boundary_supplement"][contrast.name] = boundary_interval(
            value, weights, contrast.domain, local_alpha)
        if cluster_labels is not None:
            report["cluster_sensitivity"][contrast.name] = cluster_sandwich(
                value, cube.strata, cluster_labels, local_alpha, contrast.domain, proportions=masses)
    for j, name in enumerate(cube.names):
        report["secondary"][name] = {}
        for k, field in enumerate(cube.fields):
            domain = domains.get(field)
            report["secondary"][name][field] = dict(
                mean=float(weights @ cube.means[:, j, k]), nominal_alpha=alpha,
                **_bootstrap_interval(boot["overall"][:, j, k], domain, alpha),
                boundary_supplement=None if domain is None else boundary_interval(
                    cube.means[:, j, k], weights, domain, alpha),
                by_stratum={group: float(cube.means[cube.strata == group, j, k].mean())
                            for group in groups})
    return report


def repeat_transition_counts(native, post):
    """Descriptive paired binary transitions, with no fixed repeat count."""
    native, post = np.asarray(native), np.asarray(post)
    if (native.ndim != 2 or not native.shape[0] or not native.shape[1]
            or post.shape != native.shape or not np.isin(native, [0, 1]).all()
            or not np.isin(post, [0, 1]).all()):
        raise ValueError("aligned paired binary unit-by-repeat outcomes required")
    gain, loss = (post == 1) & (native == 0), (post == 0) & (native == 1)
    zero = (post.astype(float)-native.astype(float)).mean(1) == 0
    return dict(repeat_denominator=int(native.size), units=len(native), repeats=native.shape[1],
                gain=int(gain.sum()), loss=int(loss.sum()), nochange=int((post == native).sum()),
                units_with_gain=int(gain.any(1).sum()), units_with_loss=int(loss.any(1).sum()),
                zero_unit_mean_changes=int(zero.sum()),
                zero_mean_with_offsetting_repeat_transitions=int((zero & gain.any(1) & loss.any(1)).sum()),
                all_unit_mean_changes_zero=bool(zero.all()),
                no_repeat_transitions=bool(not (gain.any() or loss.any())),
                scope="descriptive paired events; repeats are not independent units")


def run_statistics(input_path, output_dir, **options):
    """Analyze one explicit NPZ and write a new statistics.json; never overwrite."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("statistics output directory already exists; choose a new directory")
    cube = load_metrics(input_path)
    report = analyze_metrics(cube.means, cube.names, cube.fields, cube.strata, **options)
    report["input"] = dict(path=str(Path(input_path)),
                           arrays=["means", "names", "fields"] + (["strata"] if cube.strata_provided else []),
                           strata_source="input_array" if cube.strata_provided else "single_overall_group_default",
                           coordinates_read=False)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True)
    (output / "statistics.json").write_text(text, encoding="utf-8")
    return report
