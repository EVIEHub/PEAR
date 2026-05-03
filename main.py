"""
IMAD in LangGraph: a runnable skeleton

What you get:
- A LangGraph StateGraph that executes multi-agent debate over R rounds and T turns/round
- Fixed-topology and IMAD (per-round topology shuffling) supported by swapping `mode`
- Clean state object + transcript logging
- Pluggable LLM backend (stubbed by default; optional LangChain ChatOpenAI hook)

Install deps (typical):
  pip install langgraph langchain-core langchain-openai

Run:
  python imad_langgraph.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

# LangGraph
from langgraph.graph import StateGraph, END


# -----------------------------
# Types / State
# -----------------------------
Mode = Literal["fixed", "imad_uniform", "imad_subgroup", "edge_dropout"]
AggMode = Literal["majority_vote", "llm_judge"]


class Message(TypedDict):
    speaker: int
    round: int
    turn: int
    content: str


class MASState(TypedDict, total=False):
    # task
    x: str
    y_star: Optional[str]

    # config
    n_agents: int
    rounds: int
    turns_per_round: int
    mode: Mode
    agg_mode: AggMode
    base_topology: str  # "clique" | "star" | "ring" | "random_sparse"

    # stochastic control
    seed: int
    rng_state: int

    # dynamics
    round: int
    turn: int
    speaker: int
    perm: Optional[List[int]]  # permutation mapping roles->agents, 1-indexed agents
    topology: List[List[int]]  # adjacency: topology[i] = list of incoming neighbors j visible to i

    # transcript + candidates
    messages: List[Message]
    final_candidates: Dict[int, str]  # agent_id -> answer string

    # result
    decision: Optional[str]

    # logging/diagnostics
    metrics_trace: List[Dict[str, Any]]


# -----------------------------
# Topology utilities
# -----------------------------
def make_base_topology(name: str, n: int) -> List[List[int]]:
    """Return adjacency as incoming-neighbor lists (who agent i can see). 1-indexed agents."""
    if name == "clique":
        return [[j for j in range(1, n + 1) if j != i] for i in range(1, n + 1)]
    if name == "ring":
        # each i sees i-1 (wrap)
        return [[(i - 2) % n + 1] for i in range(1, n + 1)]
    if name == "star":
        # hub role = 1 in the *role graph*: everyone sees hub; hub sees everyone
        hub = 1
        topo = [[] for _ in range(n)]
        for i in range(1, n + 1):
            if i == hub:
                topo[i - 1] = [j for j in range(1, n + 1) if j != i]
            else:
                topo[i - 1] = [hub]
        return topo
    if name == "random_sparse":
        # simple fixed sparse: each node sees 2 random others
        topo = []
        for i in range(1, n + 1):
            others = [j for j in range(1, n + 1) if j != i]
            topo.append(random.sample(others, k=min(2, len(others))))
        return topo
    raise ValueError(f"Unknown base_topology={name}")


def apply_perm_to_topology(base_role_topo: List[List[int]], perm_roles_to_agents: List[int]) -> List[List[int]]:
    """
    base_role_topo is on roles 1..n (incoming neighbors in role space).
    perm_roles_to_agents maps role r -> agent perm[r-1] in 1..n.
    Return agent-space incoming neighbors: topology_agent[i] = list of agents visible to agent i.
    """
    n = len(base_role_topo)
    # invert mapping agent -> role
    agent_to_role = {agent: role for role, agent in enumerate(perm_roles_to_agents, start=1)}
    topo_agent: List[List[int]] = [[] for _ in range(n)]

    for agent_i in range(1, n + 1):
        role_i = agent_to_role[agent_i]
        incoming_roles = base_role_topo[role_i - 1]
        incoming_agents = [perm_roles_to_agents[r - 1] for r in incoming_roles]
        topo_agent[agent_i - 1] = incoming_agents

    return topo_agent


def edge_dropout(topo: List[List[int]], drop_p: float, rng: random.Random) -> List[List[int]]:
    out: List[List[int]] = []
    for incoming in topo:
        kept = [j for j in incoming if rng.random() > drop_p]
        out.append(kept)
    return out


# -----------------------------
# Views / scheduling / aggregation
# -----------------------------
def agent_view(messages: List[Message], topo: List[List[int]], agent_id: int) -> List[Message]:
    """Filter messages to those visible to agent_id under current topology (plus its own)."""
    visible_from = set(topo[agent_id - 1]) | {agent_id}
    return [m for m in messages if m["speaker"] in visible_from]


def default_speaker_schedule(n: int, turn: int) -> int:
    """Round-robin speaker schedule: 1,2,...,n,1,2,..."""
    return (turn % n) + 1


def majority_vote(cands: Dict[int, str]) -> Optional[str]:
    if not cands:
        return None
    counts: Dict[str, int] = {}
    for ans in cands.values():
        counts[ans] = counts.get(ans, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# -----------------------------
# LLM backends (stub + optional real)
# -----------------------------
@dataclass
class LLMBackend:
    """
    Minimal interface:
      generate(agent_id, prompt) -> text
    """
    generate_fn: Callable[[int, str], str]

    def generate(self, agent_id: int, prompt: str) -> str:
        return self.generate_fn(agent_id, prompt)


def make_stub_llm(seed: int) -> LLMBackend:
    rng = random.Random(seed)

    def _gen(agent_id: int, prompt: str) -> str:
        # A deterministic-ish stub: "answer" is a small hash-like choice
        options = ["A", "B", "C", "D"]
        choice = options[(hash((agent_id, prompt)) + rng.randint(0, 10_000)) % len(options)]
        return f"My answer is {choice}. Rationale: (stubbed)"
    return LLMBackend(generate_fn=_gen)


# Optional: if you want real OpenAI models via LangChain.
# Uncomment and configure OPENAI_API_KEY in env.
#
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
#
# def make_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> LLMBackend:
#     llm = ChatOpenAI(model=model, temperature=temperature)
#     def _gen(agent_id: int, prompt: str) -> str:
#         resp = llm.invoke([HumanMessage(content=prompt)])
#         return resp.content
#     return LLMBackend(generate_fn=_gen)


# -----------------------------
# LangGraph nodes
# -----------------------------
def node_init(state: MASState) -> MASState:
    rng = random.Random(state["seed"])
    state["rng_state"] = state["seed"]
    state["round"] = 1
    state["turn"] = 0
    state["speaker"] = 1
    state["messages"] = []
    state["final_candidates"] = {}
    state["decision"] = None
    state["metrics_trace"] = []
    # Precompute base role topology (roles 1..n)
    state["_base_role_topo"] = make_base_topology(state["base_topology"], state["n_agents"])  # type: ignore
    state["_rng"] = rng  # type: ignore (runtime-only)
    return state


def node_topology_scheduler(state: MASState) -> MASState:
    rng: random.Random = state["_rng"]  # type: ignore
    n = state["n_agents"]
    base_role_topo: List[List[int]] = state["_base_role_topo"]  # type: ignore

    mode = state["mode"]
    perm: Optional[List[int]] = None

    if mode == "fixed":
        # identity assignment
        perm = list(range(1, n + 1))
        topo = apply_perm_to_topology(base_role_topo, perm)
    elif mode == "imad_uniform":
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        topo = apply_perm_to_topology(base_role_topo, perm)
    elif mode == "imad_subgroup":
        # example: shuffle within 2 blocks
        perm = list(range(1, n + 1))
        mid = n // 2
        b1, b2 = perm[:mid], perm[mid:]
        rng.shuffle(b1); rng.shuffle(b2)
        perm = b1 + b2
        topo = apply_perm_to_topology(base_role_topo, perm)
    elif mode == "edge_dropout":
        perm = list(range(1, n + 1))
        topo = apply_perm_to_topology(base_role_topo, perm)
        topo = edge_dropout(topo, drop_p=0.3, rng=rng)
    else:
        raise ValueError(f"Unknown mode={mode}")

    state["perm"] = perm
    state["topology"] = topo

    # reset turn for new round
    state["turn"] = 0

    state["metrics_trace"].append({
        "event": "topology",
        "round": state["round"],
        "perm": perm,
        "topo_in_deg": [len(topo[i]) for i in range(n)],
    })
    return state


def node_turn_scheduler(state: MASState) -> MASState:
    # advance turn and pick speaker
    state["turn"] += 1
    n = state["n_agents"]
    state["speaker"] = default_speaker_schedule(n, state["turn"] - 1)
    return state


def build_agent_prompt(x: str, view: List[Message], agent_id: int) -> str:
    # Keep it simple; you’ll likely replace with your own prompt template.
    transcript = "\n".join([f"Agent {m['speaker']}: {m['content']}" for m in view[-12:]])
    return (
        "You are an agent in a multi-agent debate. Be concise and explicit.\n\n"
        f"Question:\n{x}\n\n"
        f"Visible transcript:\n{transcript}\n\n"
        "Respond with (1) your proposed final answer token (e.g., A/B/C/D or a short string), "
        "and (2) a brief justification.\n"
    )


def parse_candidate(text: str) -> str:
    # Minimal heuristic parsing: look for single-letter A-D; else take first word.
    for ch in ["A", "B", "C", "D"]:
        if f" {ch}" in f" {text}":
            return ch
    return text.strip().split()[0] if text.strip() else ""


def node_agent_step_factory(llm: LLMBackend) -> Callable[[MASState], MASState]:
    def node_agent_step(state: MASState) -> MASState:
        speaker = state["speaker"]
        topo = state["topology"]
        view = agent_view(state["messages"], topo, speaker)

        prompt = build_agent_prompt(state["x"], view, speaker)
        text = llm.generate(speaker, prompt)

        # append message
        msg: Message = {
            "speaker": speaker,
            "round": state["round"],
            "turn": state["turn"],
            "content": text,
        }
        state["messages"].append(msg)

        # update candidate
        cand = parse_candidate(text)
        if cand:
            state["final_candidates"][speaker] = cand

        state["metrics_trace"].append({
            "event": "message",
            "round": state["round"],
            "turn": state["turn"],
            "speaker": speaker,
            "cand": cand,
        })
        return state

    return node_agent_step


def node_aggregate(state: MASState) -> MASState:
    if state["agg_mode"] == "majority_vote":
        state["decision"] = majority_vote(state["final_candidates"])
    else:
        # placeholder: you can implement an LLM judge here
        state["decision"] = majority_vote(state["final_candidates"])

    state["metrics_trace"].append({
        "event": "aggregate",
        "decision": state["decision"],
        "candidates": dict(state["final_candidates"]),
    })
    return state


def node_advance_round(state: MASState) -> MASState:
    state["round"] += 1
    return state


# -----------------------------
# Routing (LangGraph conditional edges)
# -----------------------------
def route_after_turn(state: MASState) -> str:
    """After a single agent step, decide whether to schedule next turn or finalize round."""
    if state["turn"] < state["turns_per_round"]:
        return "turn"
    # round done
    if state["round"] < state["rounds"]:
        return "advance_round"
    return "aggregate"


def route_after_round_advance(state: MASState) -> str:
    return "topology"


# -----------------------------
# Build graph
# -----------------------------
def build_imad_graph(llm: LLMBackend) -> Any:
    g = StateGraph(MASState)

    g.add_node("init", node_init)
    g.add_node("topology", node_topology_scheduler)
    g.add_node("turn", node_turn_scheduler)
    g.add_node("agent", node_agent_step_factory(llm))
    g.add_node("advance_round", node_advance_round)
    g.add_node("aggregate", node_aggregate)

    g.set_entry_point("init")

    # init -> topology for round 1
    g.add_edge("init", "topology")
    # topology -> turn -> agent
    g.add_edge("topology", "turn")
    g.add_edge("turn", "agent")

    # agent -> (turn | advance_round | aggregate)
    g.add_conditional_edges("agent", route_after_turn, {
        "turn": "turn",
        "advance_round": "advance_round",
        "aggregate": "aggregate",
    })

    # advance_round -> topology
    g.add_conditional_edges("advance_round", route_after_round_advance, {
        "topology": "topology",
    })

    # aggregate -> END
    g.add_edge("aggregate", END)

    return g.compile()


# -----------------------------
# Example run
# -----------------------------
def run_one_example(
    x: str,
    *,
    n_agents: int = 5,
    rounds: int = 3,
    turns_per_round: int = 5,
    mode: Mode = "imad_uniform",
    base_topology: str = "star",
    agg_mode: AggMode = "majority_vote",
    seed: int = 0,
) -> MASState:
    llm = make_stub_llm(seed=seed)
    app = build_imad_graph(llm)

    init_state: MASState = {
        "x": x,
        "y_star": None,
        "n_agents": n_agents,
        "rounds": rounds,
        "turns_per_round": turns_per_round,
        "mode": mode,
        "base_topology": base_topology,
        "agg_mode": agg_mode,
        "seed": seed,
    }

    final_state: MASState = app.invoke(init_state)
    return final_state


if __name__ == "__main__":
    st = run_one_example(
        "Which option is correct? A) 1+1=2 B) 1+1=3 C) 1+1=4 D) 1+1=5",
        n_agents=4,
        rounds=2,
        turns_per_round=6,
        mode="imad_uniform",
        base_topology="star",
        seed=42,
    )
    print("Decision:", st.get("decision"))
    print("Final candidates:", st.get("final_candidates"))
    print("Last 3 messages:")
    for m in st["messages"][-3:]:
        print(f"  r{m['round']} t{m['turn']} agent{m['speaker']}: {m['content'][:80]}...")
    print("Trace events:", len(st["metrics_trace"]))
