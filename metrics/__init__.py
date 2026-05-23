"""Scoring and ExpPlan_v3 diagnostic metrics for AR-MAD experiments."""

from metrics.diagnostics import (
    aggregate_diagnostics,
    confidence_calibration,
    critique_acceptance_rate,
    critique_precision,
    cross_cluster_critique_rate,
    normalized_entropy,
    targeted_cross_critique_rate,
    trajectory_event_rates,
)
from metrics.scorers import accuracy, exact_match, normalize_text
from metrics.stability import bootstrap_ci, summarise_runs

__all__ = [
    "aggregate_diagnostics",
    "accuracy",
    "bootstrap_ci",
    "confidence_calibration",
    "critique_acceptance_rate",
    "critique_precision",
    "cross_cluster_critique_rate",
    "exact_match",
    "normalize_text",
    "normalized_entropy",
    "summarise_runs",
    "targeted_cross_critique_rate",
    "trajectory_event_rates",
]
