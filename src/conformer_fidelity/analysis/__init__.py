"""Statistics on complete, paired unit-level metric arrays; no experiment runner."""

from .statistics import (
    Contrast,
    MetricCube,
    analyze_metrics,
    bootstrap_means,
    boundary_interval,
    cluster_sandwich,
    hoeffding_interval,
    load_metrics,
    repeat_transition_counts,
    run_statistics,
    stratum_weights,
)

__all__ = [
    "Contrast", "MetricCube", "analyze_metrics", "bootstrap_means", "boundary_interval",
    "cluster_sandwich", "hoeffding_interval", "load_metrics", "repeat_transition_counts",
    "run_statistics", "stratum_weights",
]
