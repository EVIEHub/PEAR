# IMAD

This repo contains a minimal **LangGraph** implementation of **Itinerant Multi-Agent Debate (IMAD)**: a multi-agent debate protocol where the **communication topology is randomly shuffled each round** (i.e., agents are reassigned to network roles via a random permutation). The goal is to make debate outcomes less sensitive to arbitrary agent-role assignments (e.g., who becomes the hub in a star) and to improve stability and generalization.

---

## What’s Included

- `imad_langgraph.py` — a runnable **LangGraph** skeleton that supports:
  - **Fixed topology** (baseline)
  - **IMAD uniform shuffling** (`imad_uniform`)
  - **IMAD subgroup shuffling** (`imad_subgroup`)
  - **Edge dropout** control (`edge_dropout`)
  - A simple **majority vote** aggregator
  - A stub LLM backend (so the graph runs without external APIs)

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install langgraph langchain-core
# Optional if you plan to use OpenAI via LangChain:
pip install langchain-openai
```

> The included code uses a **stubbed LLM** by default, so it runs offline. If you want to plug in a real model, see “Using a real LLM backend”.

---

## Quickstart

Run the demo example:

```bash
python imad_langgraph.py
```

You should see output like:
- Final `Decision`
- Per-agent `Candidates`
- Number of trace events recorded

---

## How It Works (High-Level)

### Role graph vs agent graph
- You define a **base topology** as a *role graph* (e.g., star where “role 1” is hub).
- Each round, IMAD samples a permutation `perm` mapping roles → agents.
- The permuted role graph induces an **agent-space topology** (who sees whom).

### Topology-aware visibility
Each agent only sees messages from its **incoming neighbors** (plus itself). This is enforced by filtering the transcript with `agent_view(...)`.

### Debate execution
The graph runs:
- `R` rounds
- `T` turns per round (speakers scheduled round-robin)
- After the final round, an aggregator selects an answer.

---

## Configuration

In `run_one(...)`, the key knobs are:

- `n_agents`: number of agents
- `rounds`: number of rounds
- `turns_per_round`: number of turns per round
- `base_topology`: `clique | star | ring | random_sparse`
- `mode`:
  - `fixed`: no shuffling
  - `imad_uniform`: per-round uniform permutation
  - `imad_subgroup`: permute within blocks (toy example)
  - `edge_dropout`: control baseline with random edge removal
- `agg_mode`: currently `majority_vote` (LLM judge is stubbed)
- `seed`: controls randomness (permutations + stub outputs)

Example:

```python
from imad_langgraph import run_one

st = run_one(
    "Which option is correct? A) 1+1=2 B) 1+1=3 C) 1+1=4 D) 1+1=5",
    n_agents=4,
    rounds=2,
    turns_per_round=6,
    mode="imad_uniform",
    base_topology="star",
    seed=42,
)
print(st["decision"])
```

---

## Using a Real LLM Backend (Optional)

The code currently uses a stubbed backend (`make_stub_llm(seed)`), which returns deterministic-ish A/B/C/D outputs.

To use a real model, implement a backend with signature:

```python
generate(agent_id: int, prompt: str) -> str
```

and pass it into the agent node factory.

If using LangChain OpenAI:

```bash
pip install langchain-openai
export OPENAI_API_KEY="..."
```

Then implement a `make_openai_llm()` wrapper (adapt to your model/provider).

---

## Logging & Tracing

The state records:
- `messages`: transcript entries `{speaker, round, turn, content}`
- `final_candidates`: per-agent parsed answer tokens
- `metrics_trace`: event log of:
  - topology selection per round (`perm`)
  - each message event (speaker/turn)
  - aggregation event (final decision)

This is designed for:
- correctness scoring (offline)
- stability measurements (variance across permutations/seeds)
- protocol diagnostics (answer flips, consensus formation)

---

## Recommended Extensions (for real experiments)

1. Add an **LLM judge** aggregator.
2. Add a dataset runner + JSONL logging.
3. Enforce strict token budgets and fail fast on budget violations.
4. Add ablation controls (schedule shuffle only, role shuffle only, edge dropout).
5. Add stability metrics (`Δ_perm`, hub advantage, variance across seeds).

---

## License

Add your intended license here (MIT/Apache-2.0/etc.).
