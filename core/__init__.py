"""Core data structures and pure graph utilities for AR-MAD.

This subpackage contains *only* deterministic, side-effect-free building
blocks: the shared LangGraph state schema, the topology factory, the
permutation applier, and the agent-view filter. Anything that talks to the
network, an LLM, or the file system lives elsewhere.
"""

from core.state import MASState, Message, AggMode, Mode
from core.topology import (
    apply_perm_to_topology,
    edge_dropout,
    make_base_topology,
)
from core.views import agent_view, default_speaker_schedule

__all__ = [
    "MASState",
    "Message",
    "AggMode",
    "Mode",
    "make_base_topology",
    "apply_perm_to_topology",
    "edge_dropout",
    "agent_view",
    "default_speaker_schedule",
]
