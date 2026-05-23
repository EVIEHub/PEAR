"""Agent visibility filter and turn schedulers.

These helpers translate the abstract topology + counter state into the actual
*context* an agent sees on its turn. The functions are deliberately tiny so
that experiment authors can reason about them at a glance.
"""

from __future__ import annotations

import random
from typing import List

from core.state import Message
from core.topology import Adjacency


def agent_view(
    messages: List[Message],
    topo: Adjacency,
    agent_id: int,
) -> List[Message]:
    """Return the in-neighbor slice of ``messages`` visible to ``agent_id``.

    Implements the second component of the paper's observation operator
    (Eq. 5 / Eq. 9):

        Obs_i^(r)(h) := ( h_i^priv,
                          { (j, m) in h : (j -> i) in E^(r) } )

    i.e. only messages from agent_id's *incoming neighbors* under the
    *current* round's agent-space topology. The agent's own past messages
    live in :func:`agent_private` (corresponding to ``h_i^priv``) and are
    rendered separately in the prompt.

    Note that because ``topo`` is re-evaluated per round, an agent that was
    in your in-neighbor set last round but isn't this round will become
    invisible to you again -- this is the intended behaviour, and is what
    makes AR-MAD's per-round shuffle structurally meaningful.

    Parameters
    ----------
    messages:
        Full transcript so far (across all rounds).
    topo:
        Current agent-space adjacency (incoming-neighbor lists, 1-indexed).
    agent_id:
        1-indexed agent ID whose view we want.
    """
    visible = set(topo[agent_id - 1])
    return [m for m in messages if m["speaker"] in visible]


def agent_private(
    messages: List[Message],
    agent_id: int,
) -> List[Message]:
    """Return ``agent_id``'s own past messages (the ``h_i^priv`` term).

    The paper's observation operator (Eq. 5) treats the agent's own history
    as a private scratchpad that exists alongside the topology-filtered
    in-neighbor view, *not* as part of it. Keeping the two streams separate
    in the prompt makes the topology constraint legible to the agent and
    keeps the formal connection to Eq. 9 exact.
    """
    return [m for m in messages if m["speaker"] == agent_id]


def default_speaker_schedule(n: int, turn_index: int) -> int:
    """Round-robin schedule: turn 0 -> agent 1, turn 1 -> agent 2, ...

    ``turn_index`` is 0-based. Result is a 1-indexed agent ID.
    """
    return (turn_index % n) + 1


def random_speaker_schedule(n: int, rng: random.Random) -> int:
    """Pick a speaker uniformly at random.

    Used by the ``random_speaking`` mechanism-isolation control.
    """
    return rng.randint(1, n)
