"""Aggregator node: combines per-agent candidates into the final decision.

Two backends are supported:

* ``majority_vote`` -- pick the most common answer; ties broken by first-seen
  order (deterministic).
* ``llm_judge`` -- placeholder; calls an aggregator LLM with the transcript
  and the candidate set. The default implementation falls back to
  ``majority_vote`` if no judge backend is provided in ``state``.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict, Optional

from core.state import MASState
from prompts import JUDGE_TEMPLATE
from utils.logging import get_logger

_log = get_logger("nodes.aggregator")


def _majority_vote(cands: Dict[int, str]) -> Optional[str]:
    """Return the modal candidate, breaking ties by first-seen order."""
    if not cands:
        return None
    # Preserve insertion order for deterministic tie-breaking.
    first_seen: OrderedDict[str, int] = OrderedDict()
    for ans in cands.values():
        first_seen.setdefault(ans, len(first_seen))
    counts = Counter(cands.values())
    best = max(counts.items(), key=lambda kv: (kv[1], -first_seen[kv[0]]))
    return best[0]


def node_aggregate(state: MASState) -> MASState:
    """Compute ``state["decision"]`` from ``state["final_candidates"]``.

    For ``llm_judge`` we expect ``state["_judge_llm"]`` to be a
    :class:`models.base.BaseLLM` instance; if it is missing we log a
    warning and fall back to majority voting.
    """
    cands = dict(state.get("final_candidates", {}) or {})
    mode = state.get("agg_mode", "majority_vote")

    judge = state.get("judge_llm_obj")
    if mode == "llm_judge" and judge is not None:
        # Wording lives in :data:`prompts.JUDGE_TEMPLATE`; only the runtime
        # values (question text, candidate dict) are filled in here.
        prompt = JUDGE_TEMPLATE.format(question=state["x"], candidates=cands)
        gen = judge.generate(prompt, max_tokens=64)
        decision = gen.text.strip().splitlines()[0].strip() if gen.text else None
    else:
        if mode == "llm_judge":
            _log.warning("agg_mode=llm_judge but no judge LLM was provided; using majority vote.")
        decision = _majority_vote(cands)

    state["decision"] = decision
    state["metrics_trace"].append(
        {
            "event": "aggregate",
            "agg_mode": mode,
            "decision": decision,
            "candidates": cands,
        }
    )
    return state
