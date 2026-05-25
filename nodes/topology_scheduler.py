"""Init, topology-scheduling, and round-advancing nodes.

These three nodes manage the *graph-shape* state: the per-round permutation,
the agent-space adjacency, and the round counter. Everything else (turn
counters, transcripts, candidate answers) is owned by other nodes.
"""

from __future__ import annotations

from typing import List

from core.state import MASState
from core.topology import (
    apply_perm_to_topology,
    edge_dropout,
    make_base_topology,
    subgroup_permutation,
    topology_hash,
    uniform_permutation,
)
from utils.budget import Budget
from utils.logging import get_logger
from utils.seed import seeded_rng

_log = get_logger("nodes.topology")


def node_init(state: MASState) -> MASState:
    """Initialise dynamic counters and the per-run RNGs.

    Reads:
        ``seed``, ``perm_seed`` (optional), ``base_topology``, ``n_agents``.
    Writes:
        ``round``, ``turn``, ``messages``, ``final_candidates``, ``decision``,
        ``metrics_trace``, ``budget`` (and a few runtime-only underscored
        keys: ``_rng``, ``_perm_rng``, ``_base_role_topo``).
    """
    state["round"] = 1
    state["turn"] = 0
    state["speaker"] = 1
    state["messages"] = []
    state["final_candidates"] = {}
    state["decision"] = None
    state["correct"] = None
    state["metrics_trace"] = []
    state["budget"] = Budget().to_dict()

    # Two independent RNGs: one for "decoding"-style randomness (stub jitter,
    # speaker selection in ``random_speaking`` mode), one for permutations.
    # If ``perm_seed`` is not provided, fall back to ``seed`` for perms.
    state["rng"] = seeded_rng(int(state.get("seed", 0)))
    state["perm_rng"] = seeded_rng(
        int(state.get("perm_seed", state.get("seed", 0)))
    )

    # The base role topology is computed once per run. We pass the *perm* RNG
    # to ``random_sparse`` so the "fixed sparse draw" is reproducible.
    state["base_role_topo"] = make_base_topology(
        state["base_topology"],
        state["n_agents"],
        rng=state["perm_rng"],
    )

    state["metrics_trace"].append(
        {
            "event": "init",
            "n_agents": state["n_agents"],
            "rounds": state["rounds"],
            "turns_per_round": state["turns_per_round"],
            "mode": state["mode"],
            "base_topology": state["base_topology"],
            "seed": state.get("seed"),
            "perm_seed": state.get("perm_seed"),
        }
    )
    return state


def _select_perm(state: MASState) -> List[int]:
    """Pick the role->agent permutation for the current step, given mode."""
    n = state["n_agents"]
    rng = state["perm_rng"]
    mode = state["mode"]

    if mode == "fixed":
        return list(range(1, n + 1))
    if mode == "pear_uniform":
        return uniform_permutation(n, rng)
    if mode == "pear_subgroup":
        return subgroup_permutation(n, rng)
    if mode in {"edge_dropout", "random_speaking"}:
        # No permutation by construction.
        return list(range(1, n + 1))
    if mode == "role_shuffle":
        # Roles move but edges (in role-space) don't; this is functionally the
        # same as pear_uniform from the visibility standpoint, but we keep it
        # as a separate mode so trace logs can distinguish them clearly.
        return uniform_permutation(n, rng)
    raise ValueError(f"Unknown mode: {mode!r}")


def _should_reshuffle(state: MASState) -> bool:
    """Honour the ``shuffle_frequency`` configuration knob."""
    if state["mode"] == "fixed":
        return state["round"] == 1 and state.get("perm") is None
    freq = state.get("shuffle_frequency", "per_round")
    if freq == "per_round":
        return state["turn"] == 0
    if freq == "per_turn":
        return True
    if freq == "every_k_rounds":
        k = max(1, int(state.get("shuffle_every_k", 2)))
        return state["turn"] == 0 and ((state["round"] - 1) % k == 0)
    raise ValueError(f"Unknown shuffle_frequency: {freq!r}")


def node_topology_scheduler(state: MASState) -> MASState:
    """Choose ``perm`` and ``topology`` for the upcoming turn.

    The behaviour depends on ``state["mode"]`` and the ``shuffle_frequency``.
    For mechanism-isolation modes (``edge_dropout``, ``random_speaking``,
    ``role_shuffle``) we still produce a topology, but skip the permutation
    that would otherwise mix the agents.
    """
    base_role_topo = state["base_role_topo"]

    if _should_reshuffle(state) or state.get("topology") is None:
        perm = _select_perm(state)
        topo = apply_perm_to_topology(base_role_topo, perm)
        if state["mode"] == "edge_dropout":
            topo = edge_dropout(topo, drop_p=0.3, rng=state["perm_rng"])
        state["perm"] = perm
        state["topology"] = topo

        state["metrics_trace"].append(
            {
                "event": "topology",
                "round": state["round"],
                "turn": state["turn"],
                "perm": perm,
                "topology_hash": topology_hash(topo),
                "in_degree": [len(row) for row in topo],
            }
        )

    # Reset the turn counter at the start of a new round.
    if state["turn"] == 0:
        state["turn"] = 0  # explicit no-op to make the invariant obvious
    return state


def node_advance_round(state: MASState) -> MASState:
    """Increment the round counter and reset the per-round turn counter."""
    state["round"] += 1
    state["turn"] = 0
    state["metrics_trace"].append({"event": "advance_round", "round": state["round"]})
    return state
