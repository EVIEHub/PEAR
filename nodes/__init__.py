"""LangGraph node functions for the PEAR debate.

Every node takes a :class:`MASState` mapping and returns the (mutated) same
mapping. Nodes are kept as free functions (rather than methods on a class)
because LangGraph's state machine prefers callables; that also means each
node is trivially unit-testable in isolation.

Node graph (see :mod:`graph.builder`):

    init -> topology -> turn -> agent
                                  -> turn (next turn in same round)
                                  -> advance_round -> topology (next round)
                                  -> aggregate -> scorer -> END
"""

from nodes.aggregator import node_aggregate
from nodes.agent_runner import build_node_agent_step, build_agent_prompt
from nodes.scorer import build_node_scorer
from nodes.topology_scheduler import node_advance_round, node_init, node_topology_scheduler
from nodes.turn_scheduler import node_turn_scheduler

__all__ = [
    "build_agent_prompt",
    "build_node_agent_step",
    "build_node_scorer",
    "node_advance_round",
    "node_aggregate",
    "node_init",
    "node_topology_scheduler",
    "node_turn_scheduler",
]
