# Experiment Plan (LangGraph)

This plan uses **LangGraph** as the execution harness to run multi-agent debate baselines and **IMAD** (Itinerant Multi-Agent Debate: topology shuffling) under *strictly matched compute budgets*. The key idea is to keep the graph and prompts fixed while swapping only the **topology scheduler** (fixed vs shuffled).

---

## 1) Goals & Hypotheses

### Goals
- **Effectiveness:** IMAD improves task performance over fixed-topology debate at equal compute.
- **Stability:** IMAD reduces variance across seeds, agent permutations, and topology/role assignments.
- **Mechanism isolation:** Gains come from **topology shuffling (symmetry/projection)** rather than extra randomness or extra rounds/tokens.

### Hypotheses
- **H1:** IMAD increases accuracy/EM/F1 under fixed total tokens and fixed number of model calls.
- **H2:** IMAD reduces sensitivity to (i) agent relabeling, (ii) decoding seeds, and (iii) privileged positions (hub/last-speaker).
- **H3:** Improvements are largest when the base topology is asymmetric (star/chain) or agents are heterogeneous.

---

## 2) LangGraph System Design

### 2.1 State schema (per example)
Use a JSON-serializable shared state:
- `x`: task input
- `y_star`: ground truth (offline scoring only)
- `round`, `turn`: current indices
- `topology`: adjacency (incoming neighbor lists) for the current round
- `perm`: sampled permutation for IMAD
- `messages`: transcript as `[{speaker, round, turn, content}, ...]`
- `final_candidates`: per-agent candidate answers (optional)
- `decision`: final system output
- `metrics_trace`: stepwise logs (tokens, flips, agreement, judge scores, etc.)

### 2.2 Node library (reused across all conditions)
- **TopologyScheduler node**
  - Inputs: `round`, `base_graph`, `mode`, `seed`
  - Outputs: `topology`, `perm`
- **TurnScheduler node**
  - Outputs: `speaker = sigma(round, turn)`
- **AgentRunner node**
  - Builds agent prompt from `agent_view(messages, topology, agent_id)`
  - Appends message + candidate answer to state
- **Aggregator/Judge node**
  - Produces `decision` (majority vote or LLM judge)
- **Scorer node (offline)**
  - Computes metrics for the example and stores them

### 2.3 Fairness enforcement (budget control)
Hard constraints per example:
- Same `n` agents, `R` rounds, `T` turns per round
- Same max tokens per call
- Same judge model (if used)
- Same total model calls (agents + judge)

In LangGraph, enforce via:
- `metrics_trace` counters for calls/tokens
- fail-fast if a node exceeds budget

---

## 3) Conditions to Compare (swap only TopologyScheduler)

### 3.1 Fixed-topology baselines
- **Clique** (fully connected)
- **Star** (fixed hub)
- **Ring/Chain**
- **Sparse random (fixed draw)**

### 3.2 IMAD variants
- **IMAD-uniform:** per-round `perm ~ Uniform(S_n)`; topology = permuted base graph
- **IMAD-subgroup:** shuffle within blocks (team-preserving)
- **Shuffle frequency:**
  - per-round (default)
  - per-turn (strong mixing)
  - every-K-rounds (weak mixing)

### 3.3 “Random but not IMAD” controls (mechanism isolation)
- Fixed topology + **random speaking order** (schedule noise only)
- Fixed topology + **edge dropout** (same expected sparsity, no permutation averaging)
- Fixed topology + **role prompt shuffle** (roles move, edges don’t)

Expectation: IMAD should outperform these controls if symmetry/projection is the driver.

---

## 4) Benchmarks & Metrics

### 4.1 Benchmarks (suggested suite)
- **GSM8K** (Exact Match)
- **MMLU** (Accuracy)
- **HotpotQA** (EM/F1)
- **TruthfulQA** (MC Accuracy)

Use dev for tuning `n, R, T, tokens`, then freeze for test.

### 4.2 Primary metrics
- Accuracy / EM / F1 (dataset standard)

### 4.3 Stability & symmetry metrics (core)
Run repeated trials across seeds and agent relabelings:
- Mean ± std across runs
- IQR/boxplots over permutations
- Worst-case vs average-case gap
- Permutation sensitivity index:
  - `Δ_perm = E_perm[ |score(perm) − score(id)| ]`
- **Hub advantage test** (star):
  - Put best agent as hub vs leaf; IMAD should flatten the gap

### 4.4 Debate dynamics diagnostics (optional but publishable)
- Answer flip rate over turns/rounds
- Consensus/entropy over candidate answers
- Influence proxy: correlation between early messages and final decision
- Token/call counts per agent and per round

---

## 5) Replication Protocol (what to randomize)

For each condition:
- **Seeds:** 5–10 decoding seeds
- **Agent permutations:** 5–10 random relabelings of agent IDs
- **Topology randomness:** (IMAD) implicit; log `perm` each round

Total: 25–100 runs/condition depending on compute.

---

## 6) Logging Plan (LangGraph-friendly)

Per step (node execution):
- `round`, `turn`, `speaker`
- `topology_hash`, `perm`
- agent-visible context length (messages/tokens)
- candidate answer + confidence (if requested)
- judge scores (if judge)

Per example:
- final `decision`, correctness
- answer flip counts
- agreement/entropy metrics
- total calls/tokens (fairness audit)

Store as JSONL for easy aggregation.

---

## 7) Analysis & Paper-Ready Outputs

### Tables
- Main results: dataset × method (mean ± std)
- Ablations: IMAD frequency + subgroup vs uniform + random controls

### Figures
- Accuracy distribution over **agent permutations** (fixed vs IMAD)
- **Hub advantage** plot (star fixed vs IMAD-star)
- Performance vs round index (convergence / early-stop)
- Calibration plots (optional)

---

## 8) Implementation Checklist (LangGraph)

1. **BaseGraphFactory**: clique/star/ring/random_sparse (role graph)
2. **PermutationApplier**: role→agent mapping to produce agent graph
3. **AgentViewBuilder**: enforce neighbor visibility from topology
4. **LangGraph nodes**: TopologyScheduler / TurnScheduler / AgentRunner / Aggregator / Scorer
5. **Runner**:
   - loads dataset
   - executes graph
   - writes JSONL logs
   - aggregates metrics + CIs (bootstrap)

---

## 9) Minimal Core Experiment (compute-light)

- Datasets: GSM8K, MMLU, TruthfulQA
- Methods: fixed-star, fixed-clique, IMAD-star, IMAD-clique, self-consistency baseline
- Runs: 10 seeds × 10 perms on 300 questions each
- Metrics: accuracy/EM + std + `Δ_perm` + hub advantage

This is sufficient to validate the symmetry/stability thesis with clean mechanistic controls.
