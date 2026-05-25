"""Agent step: build prompt, call the LLM, append the message and candidate.

The agent is *deliberately* small: we want every prompt component to be easy
to audit, and every byte of the transcript to be reproducible from the trace
file. The actual prompt strings live in :mod:`prompts`; this module only
handles the *logic* (transcript truncation, formatting agent labels, ...).
Edit :mod:`prompts` to change phrasing without touching this file.
"""

from __future__ import annotations

from typing import Callable, Optional

from core.state import MASState, Message
from core.topology import Adjacency
from core.views import agent_private, agent_view
from models.base import BaseLLM, Generation
from prompts import AGENT_SYSTEM, AGENT_USER_TEMPLATE, EMPTY_TRANSCRIPT
from utils.budget import Budget, BudgetExceededError
from utils.logging import get_logger

_log = get_logger("nodes.agent")


def _render_messages(
    msgs: list[Message],
    *,
    window: int,
) -> str:
    """Render the most recent ``window`` messages with [r<round> t<turn>] tags.

    The round/turn prefix lets the agent see *when* a message was uttered,
    which matters under PEAR because the same speaker may have been visible
    in some previous round but not in the current one.
    """
    if not msgs:
        return EMPTY_TRANSCRIPT
    rendered = "\n".join(
        f"[r{m['round']} t{m['turn']}] Agent {m['speaker']}: {m['content']}"
        for m in msgs[-window:]
    )
    return rendered or EMPTY_TRANSCRIPT


def _build_topology_info(
    *,
    agent_id: int,
    topo: Adjacency,
    n_agents: int,
    round_idx: int,
    turn_idx: int,
) -> str:
    """Compose the per-turn topology summary block.

    Tells the agent (a) which round and turn this is, (b) which agent IDs
    are currently in its in-neighbour set (whose messages it can read),
    and (c) which agents have it as an in-neighbour (so its forthcoming
    reply will be visible to them). Knowing this on the LLM side keeps
    the agent honest about PEAR's per-round changing topology without
    leaking any additional content.
    """
    in_neigh = sorted(topo[agent_id - 1])
    out_neigh = sorted(j for j in range(1, n_agents + 1) if agent_id in topo[j - 1])
    in_str = str(in_neigh) if in_neigh else "(none -- only your own private history)"
    out_str = str(out_neigh) if out_neigh else "(no one this round)"
    return (
        f"Round {round_idx}, turn {turn_idx}.\n"
        f"In-neighbours visible to you this round: {in_str}.\n"
        f"Your reply will be observable to: {out_str}.\n\n"
    )


def build_agent_prompt(
    x: str,
    view_neighbors: list[Message],
    view_private: list[Message],
    agent_id: int,
    *,
    topo: Adjacency,
    n_agents: int,
    round_idx: int,
    turn_idx: int,
    transcript_window: int = 12,
) -> str:
    """Assemble the user-side prompt for one agent turn (topology-aware).

    Two visible streams are rendered separately to mirror the paper's
    observation operator (Eq. 5 / Eq. 9):

    * ``view_private``   -- the agent's own past messages (``h_i^priv``).
    * ``view_neighbors`` -- messages from the agent's *current* in-neighbours
      under the active round's topology, i.e.
      ``{ (j, m) in h : (j -> i) in E^(r) }``.

    The two streams plus a small ``topology_info`` block are interpolated
    into :data:`prompts.AGENT_USER_TEMPLATE`. Edit that template to tweak
    wording without touching this function.
    """
    private = _render_messages(view_private, window=transcript_window)
    transcript = _render_messages(view_neighbors, window=transcript_window)
    topology_info = _build_topology_info(
        agent_id=agent_id,
        topo=topo,
        n_agents=n_agents,
        round_idx=round_idx,
        turn_idx=turn_idx,
    )
    return AGENT_USER_TEMPLATE.format(
        agent_id=agent_id,
        question=x,
        topology_info=topology_info,
        private=private,
        transcript=transcript,
    )


def build_agent_call(state: MASState) -> tuple[int, str, int]:
    """Return ``(speaker, prompt, max_tokens)`` for the current agent turn."""
    speaker = state["speaker"]
    topo = state["topology"]
    view_neighbors = agent_view(state["messages"], topo, speaker)
    view_private = agent_private(state["messages"], speaker)
    prompt = build_agent_prompt(
        state["x"],
        view_neighbors,
        view_private,
        speaker,
        topo=topo,
        n_agents=int(state["n_agents"]),
        round_idx=int(state["round"]),
        turn_idx=int(state["turn"]),
    )
    return speaker, prompt, int(state.get("max_tokens_per_call", 256))


def apply_agent_generation(
    state: MASState,
    gen: Generation,
    *,
    parse_answer: Callable[[str], str],
) -> MASState:
    """Apply one completed agent generation to the mutable state."""
    speaker = state["speaker"]

    budget_dict = state.get("budget") or {}
    budget = Budget(
        max_calls=budget_dict.get("max_calls", 0),
        max_tokens=budget_dict.get("max_tokens", 0),
        calls=budget_dict.get("calls", 0),
        prompt_tokens=budget_dict.get("prompt_tokens", 0),
        completion_tokens=budget_dict.get("completion_tokens", 0),
    )
    try:
        budget.charge(
            calls=1,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )
    except BudgetExceededError:
        _log.error("Budget exceeded; aborting run.")
        raise
    state["budget"] = budget.to_dict()

    msg: Message = {
        "speaker": speaker,
        "round": state["round"],
        "turn": state["turn"],
        "content": gen.text,
    }
    state["messages"].append(msg)

    candidate = parse_answer(gen.text)
    if candidate:
        state["final_candidates"][speaker] = candidate

    prev = None
    for m in reversed(state["messages"][:-1]):
        if m["speaker"] == speaker:
            prev = parse_answer(m["content"])
            break

    state["metrics_trace"].append(
        {
            "event": "message",
            "round": state["round"],
            "turn": state["turn"],
            "speaker": speaker,
            "candidate": candidate,
            "flipped": bool(prev and candidate and prev != candidate),
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "content": gen.text,
        }
    )
    return state


def build_node_agent_step(
    llm: BaseLLM,
    *,
    system_prompt: Optional[str] = AGENT_SYSTEM,
    parse_answer: Optional[Callable[[str], str]] = None,
) -> Callable[[MASState], MASState]:
    """Return a node function that runs one LLM call for the current speaker.

    Parameters
    ----------
    llm:
        Backend implementing :class:`models.base.BaseLLM`.
    system_prompt:
        Optional system instruction.
    parse_answer:
        Callable mapping completion text to the candidate answer token. If
        omitted, we fall back to a tiny heuristic that prefers the shared JSON
        ``answer`` field and then older answer-line formats.

    Returns
    -------
    Callable[[MASState], MASState]
        A LangGraph-compatible node function.
    """

    if parse_answer is None:
        parse_answer = _default_parse

    def node_agent_step(state: MASState) -> MASState:
        speaker, prompt, max_tokens = build_agent_call(state)
        gen = llm.generate(
            prompt,
            max_tokens=max_tokens,
            system=system_prompt,
            agent_id=speaker,
        )
        return apply_agent_generation(state, gen, parse_answer=parse_answer)

    return node_agent_step


def _default_parse(text: str) -> str:
    """Very simple fallback parser for completions.

    Prefers a JSON ``answer`` field; otherwise returns an older explicit
    answer line or the first A-D letter found in the text.
    """
    if not text:
        return ""
    import json
    import re

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("answer") is not None:
            return str(payload["answer"]).strip()

    m = re.search(r"final\s*answer\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".").strip()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else ""
