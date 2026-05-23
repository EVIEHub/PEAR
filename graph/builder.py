"""Compile the AR-MAD :class:`StateGraph`.

The graph topology is the same across every condition; the conditions differ
*only* in what the topology scheduler produces, which is exactly the design
goal stated in ExpPlan.md (swap the scheduler, leave everything else fixed).

::

    init -> sched_topology -> sched_turn -> agent
                                              |-> sched_turn  (more turns this round)
                                              |-> advance_round -> sched_topology
                                              '-> aggregate -> scorer -> END

Note on naming: LangGraph (>=0.2) refuses to register a node whose name
shadows a key in the state TypedDict. Our state already uses ``topology``
and ``turn`` as channels, so the corresponding scheduler nodes are named
``sched_topology`` and ``sched_turn`` to avoid the collision.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from core.state import MASState
from models.base import BaseLLM
from nodes import (
    build_node_agent_step,
    build_node_scorer,
    node_advance_round,
    node_aggregate,
    node_init,
    node_topology_scheduler,
    node_turn_scheduler,
)


def _route_after_turn(state: MASState) -> str:
    """Decide whether to schedule another turn, advance the round, or finalize."""
    if state["turn"] < state["turns_per_round"]:
        return "sched_turn"
    if state["round"] < state["rounds"]:
        return "advance_round"
    return "aggregate"


def _route_after_round_advance(state: MASState) -> str:
    """Trivially route to the topology scheduler after a round boundary."""
    return "sched_topology"


def build_armad_graph(
    llm: BaseLLM,
    *,
    judge_llm: Optional[BaseLLM] = None,
    parse_answer=None,
) -> Any:
    """Compile and return a runnable LangGraph application.

    Parameters
    ----------
    llm:
        Backend used by every agent.
    judge_llm:
        Optional separate backend for the LLM-judge aggregator. If provided,
        the experiment runner should attach it to the state under the
        runtime-only key ``"judge_llm_obj"`` before invoking the graph.
    parse_answer:
        Optional task-specific answer parser, forwarded to the agent node.

    Returns
    -------
    A compiled LangGraph application with ``invoke(state)`` semantics.
    """
    g = StateGraph(MASState)

    g.add_node("init", node_init)
    g.add_node("sched_topology", node_topology_scheduler)
    g.add_node("sched_turn", node_turn_scheduler)
    g.add_node("agent", build_node_agent_step(llm, parse_answer=parse_answer))
    g.add_node("advance_round", node_advance_round)
    g.add_node("aggregate", node_aggregate)
    g.add_node("scorer", build_node_scorer())

    g.set_entry_point("init")

    g.add_edge("init", "sched_topology")
    g.add_edge("sched_topology", "sched_turn")
    g.add_edge("sched_turn", "agent")

    g.add_conditional_edges(
        "agent",
        _route_after_turn,
        {
            "sched_turn": "sched_turn",
            "advance_round": "advance_round",
            "aggregate": "aggregate",
        },
    )
    g.add_conditional_edges(
        "advance_round",
        _route_after_round_advance,
        {"sched_topology": "sched_topology"},
    )
    g.add_edge("aggregate", "scorer")
    g.add_edge("scorer", END)

    return g.compile()
