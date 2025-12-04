```python
User ─→ CLI/Channel ─→ Agent ─→ LLM Backend
                       │
                       ├─ Policy Layer
                       ├─ Δφ Topology Engine
                       └─ Trace/Runtime
```

🚀 Operor Multi-Agent Package

Delta-phi Topology × Multi-Channel Runtime Architecture


---

✨ Overview

Operor Multi-Agent Package is a next-generation framework for building multi-agent LLM systems.
It integrates a mathematically interpretable Δφ (Delta-phi) topology layer with fully isolated multi-agent runtimes, enabling multiple agents to run in parallel without state leakage.

This package is the implementation backbone of the Sofience–Operor Engine, providing:

📡 Parallel agent execution

📈 Δφ-based reasoning evolution tracking

🔒 Runtime & memory isolation

🧭 Hybrid policy alignment

🧩 Multi-channel LLM orchestration

🛠 CI-validated stability



---

🌐 Core Concepts

🔷 Δφ Topology Layer

A formalism for expressing how reasoning changes across steps:

Δφ magnitude

Δφ severity

Δφ propagation

φ-trajectory clustering

Void-delta computation


This produces a measurable, interpretable “phase-shift vector” representing how an agent's reasoning evolves.


---

🔶 Multi-Agent Runtime Isolation

Every agent gets its own:

OperorRuntime

TraceLog

PhaseState history

Environment deltas


This guarantees that even when sharing a backend LLM,
their cognitive trajectories remain fully independent.


---

🟦 Multi-Channel Architecture

Each agent uses multiple internal channels:

Planning channel

Semantic reasoning channel

Policy refinement channel


These channels run in parallel and then get aggregated into a single coherent output.


---

🔧 Features

✔ Parallel multi-channel LLM execution

✔ Δφ propagation engine

✔ Runtime isolation per agent

✔ Hybrid policy system (keyword/semantic)

✔ Observability hooks for Δφ, Void, φ-trajectory

✔ Structured TraceLog per step

✔ GitHub Actions CI

✔ pytest coverage for Δφ and runtime isolation



---

🧪 Test Coverage

CI runs four key test suites:

1) Basic agent-step execution

Ensures each call produces a coherent multi-channel output.

2) Trace accumulation

Sequential calls must increase the trace length.

3) Δφ propagation

Confirms Δφ(magnitude/severity) changes as environment deltas shift.

4) Multi-agent runtime isolation

Creates three runtimes and verifies:

Independent Δφ histories

No overlapping trace IDs

No cross-runtime pollution

Valid output from each agent


Example CI output:

```python 
============================= test session starts =============================
collected 4 items

tests/test_agent_step.py ....
============================== 4 passed in 0.04s ==============================
```

---

🏗 Architecture Diagram

```python 
┌─────────────────────────────────────────┐
│         Operor Multi-Agent Engine       │
├─────────────────────────────────────────┤
│        Multi-Channel Agent Layer        │
│     ├─ PlannerAgent                     │
│     ├─ SemanticAgent                    │
│     └─ PolicyAgent                      │
├─────────────────────────────────────────┤
│       Runtime Layer (isolated state)    │
│     ├─ OperorRuntime                    │
│     ├─ TraceLog (Δφ history)            │
│     └─ Environment states               │
├─────────────────────────────────────────┤
│            Δφ Topology Layer            │
│     ├─ Δφ magnitude                     │
│     ├─ Δφ severity                      │
│     └─ Propagation engine               │
├─────────────────────────────────────────┤
│       Observability / Debug Hooks       │
└─────────────────────────────────────────┘
```

---

🚦 Quick Start

from operor_multi_agent import agent_step, OperorRuntime

runtime = OperorRuntime()

reply = agent_step(
    "Summarize my tasks for today.",
    env_state={"need_level": 0.7, "supply_level": 0.2},
    runtime=runtime,
)

print(reply)
print(runtime.trace_log.entries[-1].delta_phi_vec)


---

📈 Roadmap

Completed

✔ Core multi-agent architecture

✔ Δφ topology layer

✔ Hybrid policy alignment

✔ Runtime isolation

✔ Observability hooks

✔ Full GitHub Actions CI

✔ pytest: Δφ + multi-agent isolation


Upcoming

⏳ Async multi-agent execution

⏳ FastAPI interface

⏳ Long-term memory / RAG integration

⏳ Tool-use / function-calling support

⏳ Kubernetes deployment

⏳ Δφ visualization dashboard



---

🧭 Vision

The Operor Multi-Agent Package is not just a demonstration.
It is a structural experiment proposing a new way to define state, phase change, and parallel reasoning in LLM systems.

The Δφ Formalism provides a mathematically interpretable lens into how reasoning evolves.
The Operor Runtime ensures independent cognitive trajectories for each agent.

Together, they form the foundation for next-wave LLM system architecture.


---

❤️ Acknowledgements

This project is developed with conceptual support from:

Sofience

Δφ (Delta-phi) Formalism

Operor-based multi-agent reasoning models
준비 중
