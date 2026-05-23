"""Scorer node: computes per-example correctness against ground truth.

The scorer is task-aware via the :class:`data.tasks.Task` object that
the experiment runner supplies in ``state["_task"]`` (runtime-only key).
That keeps the node trivially generic across benchmarks: GSM8K's numeric
EM, MMLU's letter match, HotpotQA's lower-cased EM all flow through
``Task.score(prediction, example)``.
"""

from __future__ import annotations

from typing import Callable

from core.state import MASState
from utils.logging import get_logger

_log = get_logger("nodes.scorer")


def build_node_scorer() -> Callable[[MASState], MASState]:
    """Return a scorer node that uses ``state["_task"]`` and ``state["_example"]``.

    We expose this as a *factory* (rather than a free function) for symmetry
    with the agent node and so future versions can close over additional
    scoring helpers (e.g. a tokenizer for F1) without changing the signature.
    """

    def node_scorer(state: MASState) -> MASState:
        task = state.get("task_obj")
        ex = state.get("example_obj")
        decision = state.get("decision")

        if task is None or ex is None:
            # Offline scoring is optional: leave correctness as ``None``.
            state["correct"] = None
            return state

        prediction = task.parse_answer(decision or "")
        # Some aggregators already return a parsed token; only re-parse when
        # the decision string still looks like full free-form text.
        if not prediction:
            prediction = decision or ""
        ok = task.score(prediction, ex)
        state["correct"] = bool(ok)
        state["metrics_trace"].append(
            {
                "event": "score",
                "prediction": prediction,
                "gold": ex.answer,
                "correct": bool(ok),
            }
        )
        return state

    return node_scorer
