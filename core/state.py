"""Shared state schema for the AR-MAD LangGraph.

The graph is a single ``StateGraph`` whose nodes mutate a single dictionary
that conforms to ``MASState``. Every channel that nodes read or write must
be declared here, including the runtime-only objects that are not safe to
serialize (random generators, the live :class:`Task` instance, etc.); the
JSONL tracer drops those keys via :data:`RUNTIME_ONLY_KEYS` before writing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# Type aliases for the small finite enumerations used in the graph.
Mode = Literal[
    "fixed",            # No shuffling, identity permutation.
    "armad_uniform",     # Per-step uniform permutation over S_n.
    "armad_subgroup",    # Permutation only within blocks (team-preserving).
    "edge_dropout",     # Mechanism-isolation control: drop random edges.
    "random_speaking",  # Mechanism-isolation control: schedule noise only.
    "role_shuffle",     # Mechanism-isolation control: roles move, edges don't.
]
"""Discrete topology / scheduling modes from ExpPlan.md sections 3.1-3.3."""

AggMode = Literal["majority_vote", "llm_judge"]
"""How the final answer is aggregated from per-agent candidates."""


class Message(TypedDict):
    """One transcript entry, produced by an :class:`AgentRunner` step."""

    speaker: int   # 1-indexed agent ID who emitted the message.
    round: int     # 1-indexed round number.
    turn: int      # 1-indexed turn within the round.
    content: str   # Raw model output text.


class MASState(TypedDict, total=False):
    """LangGraph state shared by every node in the AR-MAD graph.

    All keys are optional in the type sense (``total=False``) because nodes
    populate them incrementally; in practice ``init`` is responsible for
    establishing every required field before the first scheduler runs.

    Attributes
    ----------
    x:
        The task input string presented to the agents (e.g. a GSM8K question).
    y_star:
        Optional ground-truth answer. Used by the offline scorer; never shown
        to agents.
    n_agents, rounds, turns_per_round:
        Debate hyper-parameters.
    mode, agg_mode, base_topology:
        Discrete configuration choices, see :data:`Mode` and :data:`AggMode`.
    seed, rng_state:
        ``seed`` is the user-controlled random seed; ``rng_state`` is reserved
        for future use if the run needs to be resumed deterministically.
    round, turn, speaker:
        Live counters maintained by the schedulers.
    perm:
        Current role-to-agent permutation (1-indexed). ``perm[r-1] = a`` means
        role ``r`` is currently played by agent ``a``.
    topology:
        Adjacency in *agent space*. ``topology[i-1]`` is the list of agents
        whose messages agent ``i`` is allowed to read this round.
    messages, final_candidates:
        Running transcript and per-agent latest answer token.
    decision:
        Aggregator output once the debate finishes.
    correct:
        Boolean correctness, written by the scorer node.
    metrics_trace:
        Append-only event log for offline analysis.
    budget:
        Running counters for fairness audit (calls / tokens). See
        :mod:`utils.budget`.
    """

    # Task
    x: str
    y_star: Optional[str]

    # Static configuration
    n_agents: int
    rounds: int
    turns_per_round: int
    mode: Mode
    agg_mode: AggMode
    base_topology: str
    max_tokens_per_call: int
    shuffle_frequency: str
    shuffle_every_k: int

    # Stochastic control
    seed: int
    perm_seed: int
    rng_state: int

    # Dynamic counters
    round: int
    turn: int
    speaker: int
    perm: Optional[List[int]]
    topology: List[List[int]]

    # Transcript and per-agent state
    messages: List[Message]
    final_candidates: Dict[int, str]

    # Result and diagnostics
    decision: Optional[str]
    correct: Optional[bool]
    metrics_trace: List[Dict[str, Any]]
    budget: Dict[str, int]

    # Per-agent metadata (optional). Index by agent_id (1-indexed).
    agent_labels: Dict[int, str]

    # Runtime-only channels
    # Declared here so LangGraph forwards them between nodes; *not* JSON-
    # serializable, so the JSONL tracer filters them out via
    # :data:`RUNTIME_ONLY_KEYS` below.
    rng: Any                      # decoding / scheduling RNG (random.Random)
    perm_rng: Any                 # permutation RNG (random.Random)
    base_role_topo: List[List[int]]
    task_obj: Any                 # data.tasks.Task instance
    example_obj: Any              # data.tasks.Example instance
    judge_llm_obj: Any            # models.base.BaseLLM instance


#: Keys that the JSONL tracer must drop before writing to disk.
RUNTIME_ONLY_KEYS: tuple[str, ...] = (
    "rng",
    "perm_rng",
    "base_role_topo",
    "task_obj",
    "example_obj",
    "judge_llm_obj",
)
