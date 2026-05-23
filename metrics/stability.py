"""Stability metrics from ExpPlan.md section 4.3.

The functions here operate on the *aggregated results* of multiple runs,
typically the JSONL files written under ``outputs/<run>/results.jsonl``.

Implemented metrics
-------------------
* :func:`summarise_runs` -- mean ± std + IQR + min/max for an iterable of
  numeric scores.
* :func:`bootstrap_ci` -- non-parametric bootstrap confidence interval.
* :func:`delta_perm` -- permutation-sensitivity index
  ``E_perm |score(perm) - score(id)|``.
* :func:`hub_advantage` -- gap between best vs worst hub assignments under
  a star topology.
* :func:`position_role_gap` -- generalisation of :func:`hub_advantage` to
  any structural-role taxonomy (first speaker, ring index, ...).
* :func:`worst_case`, :func:`percentile` -- tail-risk summaries that
  complement the mean (AR-MAD typically wins on tails even when means tie).
* :func:`outcome_invariance` -- agreement between paired runs that only
  differ in an agent-ID relabelling (permutation-equivariance probe).
* :func:`poisoning_drop` -- accuracy drop after injecting a
  confident-wrong agent into a privileged position.
* :func:`early_anchor_rate` -- fraction of runs whose final decision
  matches the turn-1 majority (high = early speakers dominate).
* :func:`diversity_curve` -- per-round entropy of the candidate-answer
  distribution; flags premature convergence / echo-chamber dynamics.
* :func:`influence_gini` -- Gini coefficient of per-agent influence on the
  final decision; low = balanced contribution across agents.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, pstdev, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


# Summary statistics
@dataclass
class RunSummary:
    """Compact summary statistics for a sequence of scalar scores."""

    n: int
    mean: float
    std: float
    iqr: Tuple[float, float]
    min: float
    max: float
    raw: List[float] = field(default_factory=list)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile on a pre-sorted sequence."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def summarise_runs(scores: Iterable[float]) -> RunSummary:
    """Compute mean / std / IQR / min / max for a sequence of scores."""
    values = [float(s) for s in scores]
    if not values:
        return RunSummary(0, float("nan"), float("nan"), (float("nan"), float("nan")),
                          float("nan"), float("nan"), [])
    sorted_v = sorted(values)
    return RunSummary(
        n=len(values),
        mean=mean(values),
        std=stdev(values) if len(values) > 1 else 0.0,
        iqr=(_quantile(sorted_v, 0.25), _quantile(sorted_v, 0.75)),
        min=min(values),
        max=max(values),
        raw=values,
    )


# Bootstrap
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the *mean* of ``values``."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_q = (1 - confidence) / 2
    hi_q = 1 - lo_q
    return (_quantile(means, lo_q), _quantile(means, hi_q))


# AR-MAD-specific stability metrics
def delta_perm(
    scores_by_perm: Mapping[Tuple[int, ...], float],
    identity: Tuple[int, ...] | None = None,
) -> float:
    """Permutation-sensitivity index ``E_perm[ |score(perm) - score(id)| ]``.

    Parameters
    ----------
    scores_by_perm:
        Mapping from a permutation (as a tuple) to a scalar score (e.g.
        accuracy on the eval set).
    identity:
        Permutation considered "the identity"; defaults to the sorted tuple
        of agent IDs inferred from the keys.

    Returns
    -------
    float
        The mean absolute deviation from the identity score. Returns NaN if
        the input is empty or the identity is missing.
    """
    if not scores_by_perm:
        return float("nan")
    if identity is None:
        any_key = next(iter(scores_by_perm))
        identity = tuple(sorted(any_key))
    if identity not in scores_by_perm:
        return float("nan")
    base = scores_by_perm[identity]
    deltas = [abs(s - base) for s in scores_by_perm.values()]
    return sum(deltas) / len(deltas)


def hub_advantage(
    scores_by_hub: Mapping[int, float],
) -> float:
    """Difference between best and worst hub assignment under a star.

    Parameters
    ----------
    scores_by_hub:
        ``{agent_id: score}``, where ``agent_id`` is the agent that played
        the hub role in that experimental cell.

    Returns
    -------
    float
        ``max(scores) - min(scores)``. A *small* number indicates that
        AR-MAD has flattened the hub bias.
    """
    if not scores_by_hub:
        return float("nan")
    values = list(scores_by_hub.values())
    return max(values) - min(values)


def position_role_gap(
    scores_by_role: Mapping[Any, float],
) -> float:
    """Generalised :func:`hub_advantage` over an arbitrary role taxonomy.

    ``scores_by_role`` keys can be any structural label -- ``"first_speaker"``
    / ``"last_speaker"`` for chains, ring indices, or tuples encoding
    composite roles. The metric is again ``max - min``: small values mean
    the method is approximately invariant to which agent occupies which
    structural position, which is the symmetry property AR-MAD targets.
    """
    if not scores_by_role:
        return float("nan")
    values = [float(v) for v in scores_by_role.values()]
    return max(values) - min(values)


# Tail-risk summaries
def worst_case(scores: Iterable[float]) -> float:
    """Minimum score across runs (lower bound under unfavourable draws).

    For a fixed-topology MAD baseline this corresponds to the unluckiest
    permutation; for AR-MAD it is the worst stochastic run. Reporting both
    next to the mean is the cleanest way to surface AR-MAD's tail-robustness
    advantage when means are roughly comparable.
    """
    values = [float(s) for s in scores]
    if not values:
        return float("nan")
    return min(values)


def percentile(scores: Iterable[float], q: float) -> float:
    """Linearly-interpolated ``q``-quantile of ``scores`` (``q`` in [0, 1]).

    Useful as a smoother tail summary than :func:`worst_case`: e.g. the 5th
    or 10th percentile of accuracy across permutations.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    values = sorted(float(s) for s in scores)
    if not values:
        return float("nan")
    return _quantile(values, q)


# Symmetry / robustness probes
def outcome_invariance(
    decisions_a: Sequence[Any],
    decisions_b: Sequence[Any],
) -> float:
    """Fraction of paired examples on which two runs returned the same answer.

    The intended pairing is "same example, same seed, but agent IDs are
    relabelled by some permutation between run A and run B". A
    permutation-equivariant method (AR-MAD in expectation) will match on
    every pair (returns 1.0); a fixed-topology method whose decision
    depends on positional roles will diverge whenever the relabelling
    moves an influential role onto a different agent.

    Returns NaN if the two sequences are empty or have different lengths.
    """
    if not decisions_a or len(decisions_a) != len(decisions_b):
        return float("nan")
    matches = sum(1 for a, b in zip(decisions_a, decisions_b) if a == b)
    return matches / len(decisions_a)


def poisoning_drop(acc_clean: float, acc_poisoned: float) -> float:
    """Accuracy drop induced by a confident-wrong agent injection.

    ``acc_clean`` is the baseline accuracy without injection; ``acc_poisoned``
    is the accuracy when one agent is forced to argue for a wrong answer
    (typically while occupying the most privileged role: hub in a star,
    first speaker in a chain, ...). A *small* drop means the method
    tolerates an adversarial peer well -- AR-MAD should drop less because
    rotation prevents the bad agent from monopolising any role.

    Reported as ``acc_clean - acc_poisoned`` so larger values mean *worse*
    robustness; positive convention matches the rest of the module.
    """
    return float(acc_clean) - float(acc_poisoned)


# Debate-dynamics diagnostics
def early_anchor_rate(
    turn1_majority: Sequence[Any],
    final_decision: Sequence[Any],
) -> float:
    """Fraction of runs whose final decision matches the turn-1 majority.

    A high value indicates that whoever spoke first effectively decided the
    outcome (anchoring / cascade). AR-MAD's per-round shuffle is supposed to
    re-route influence through later rounds, so its anchor rate should be
    materially lower than fixed-topology MAD on the same problem.

    Both inputs must have the same length; ``None`` entries are treated as
    "no decision" and never count as a match.
    """
    if not turn1_majority or len(turn1_majority) != len(final_decision):
        return float("nan")
    matches = 0
    for early, final in zip(turn1_majority, final_decision):
        if early is None or final is None:
            continue
        if early == final:
            matches += 1
    return matches / len(turn1_majority)


def diversity_curve(
    candidates_by_round: Sequence[Sequence[Any]],
) -> List[float]:
    """Per-round Shannon entropy (in bits) of the candidate-answer set.

    ``candidates_by_round[r]`` should be the list of per-agent candidates
    observed at the end of round ``r`` (use parsed tokens, not free-form
    text). The returned list has the same length; rounds whose entries are
    all empty / ``None`` contribute 0.0.

    Use this to plot how quickly the agents collapse onto a single answer.
    Fixed-topology debate often collapses by round 2; healthy AR-MAD runs
    should hold non-zero entropy for one or two extra rounds before
    converging, which is empirical evidence that shuffling delays
    premature consensus.
    """
    out: List[float] = []
    for cands in candidates_by_round:
        cleaned = [c for c in cands if c is not None and c != ""]
        if not cleaned:
            out.append(0.0)
            continue
        counts = Counter(cleaned)
        total = sum(counts.values())
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * math.log2(p)
        out.append(h)
    return out


def influence_gini(per_agent_hits: Mapping[int, float]) -> float:
    """Gini coefficient over per-agent influence on the final decision.

    ``per_agent_hits[i]`` is typically the count (or frequency) of examples
    whose system-level decision equals agent ``i``'s terminal candidate;
    other proxies (e.g. citation counts in transcripts) work too as long as
    they are non-negative.

    A Gini of 0 means every agent contributes equally; values near 1 mean
    a single agent monopolises the outcome -- the latter is exactly the
    ``hub`` / ``first-speaker`` failure mode that AR-MAD is supposed to
    flatten. Returns NaN on empty input and 0.0 if the sum of hits is 0.
    """
    values = sorted(float(v) for v in per_agent_hits.values())
    n = len(values)
    if n == 0:
        return float("nan")
    total = sum(values)
    if total <= 0:
        return 0.0
    weighted = sum(i * v for i, v in enumerate(values, start=1))
    # Standard Gini on a sorted non-negative sample.
    return (2.0 * weighted) / (n * total) - (n + 1) / n