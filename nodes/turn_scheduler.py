"""Per-turn scheduler: increments the turn counter and picks a speaker."""

from __future__ import annotations

from core.state import MASState
from core.views import default_speaker_schedule, random_speaker_schedule


def node_turn_scheduler(state: MASState) -> MASState:
    """Advance ``turn`` by 1 and select the speaker for this turn.

    The speaker policy depends on ``state["mode"]``:

    * ``random_speaking`` -- uniform random over agents (mechanism-isolation
      control: schedule noise without topology shuffling).
    * Everything else -- round-robin in agent ID order.

    The speaker IDs are 1-indexed.
    """
    state["turn"] += 1
    n = state["n_agents"]
    if state.get("mode") == "random_speaking":
        rng = state["rng"]
        state["speaker"] = random_speaker_schedule(n, rng)
    else:
        state["speaker"] = default_speaker_schedule(n, state["turn"] - 1)
    return state
