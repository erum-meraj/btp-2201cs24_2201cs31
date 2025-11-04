# BTP- 2201CS24 2201CS31

Authors:

- Erum Meraj @erum-meraj
- Hrishikesh Choudhary @vivekananda1001

Agentic workflow framework for multi-task offloading (edge/cloud). This repository composes following modules:

- planner (LLM-guided plan generation)
- evaluator (cost model + heuristic search)
- output (LLM-based explanation)
- core simulation (network, environment, workflow, cost model)
- runner / UI (Streamlit chat demo)

Project layout

- [agents/](agents) — agent implementations and orchestration
  - [agents/main.py](agents/main.py)
  - [agents/base_agent.py](agents/base_agent.py)
  - [agents/planner.py](agents/planner.py)
  - [agents/evaluator.py](agents/evaluator.py)
  - [agents/output.py](agents/output.py)
  - [agents/result.md](agents/result.md)
- [core/](core) — core data types, environment and cost evaluation
  - [core/workflow.py](core/workflow.py)
  - [core/network.py](core/network.py)
  - [core/environment.py](core/environment.py)
  - [core/cost_eval.py](core/cost_eval.py)
  - [core/utils.py](core/utils.py)
- [runner/run_graph.py](runner/run_graph.py) — simple example runner using StateGraph
- [chat_interface.py](chat_interface.py) — Streamlit chat UI demo
- [.env](.env), [requirements.txt](requirements.txt), [pyproject.toml](pyproject.toml)

Key components (symbols)

- Planner agent: [`agents.planner.PlannerAgent`](agents/planner.py)
- Evaluator agent: [`agents.evaluator.EvaluatorAgent`](agents/evaluator.py)
- Output agent: [`agents.output.OutputAgent`](agents/output.py)
- Base LLM wrapper: [`agents.base_agent.BaseAgent`](agents/base_agent.py)
- Workflow and Task types: [`core.workflow.Workflow`](core/workflow.py), [`core.workflow.Task`](core/workflow.py)
- Network model: [`core.network.Network`](core/network.py), [`core.network.Node`](core/network.py), [`core.network.Link`](core/network.py)
- Environment model: [`core.environment.Environment`](core/environment.py)
- Utility cost evaluator: [`core.cost_eval.UtilityEvaluator`](core/cost_eval.py)
- Topological sort helper: [`core.utils.topological_sort`](core/utils.py)
- Runner helper: [`runner.run_graph.run_experiment`](runner/run_graph.py)

What the system computes

- The evaluator uses a cost model implemented in [`core.cost_eval.UtilityEvaluator`](core/cost_eval.py).
  The total offloading cost is defined as
  $$
  U(w,p)=\delta_t \cdot T + \delta_e \cdot E
  $$
  where $T$ is a delay metric (critical-path delay) and $E$ is an energy-based cost (see file for details).

Quick start

1. Create virtual environment and install deps

```bash
python -m venv .venv
# Activate .venv (platform-specific)
pip install -r requirements.txt
```

2. Add your Google API key to `.env`:

```text
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

(see [.env](.env) for format)

3. Run the Streamlit chat UI (recommended)

```bash
streamlit run chat_interface.py
```

This launches the interactive interface that uses the agentic workflow implemented in [agents/main.py](agents/main.py).

4. run the agents workflow as a package:
   From repository root run:

```bash
python -m agents.main
```

(Prefer running as a module to ensure sibling-package imports like `core` resolve correctly.)

Development notes

- Cost internals located in [`core/cost_eval.py`](core/cost_eval.py) — extend or tune CT / CE / delta parameters in [`agents/evaluator.EvaluatorAgent.find_best_policy`](agents/evaluator.py) when calling the evaluator.
- To change LLM model, edit [`agents/base_agent.BaseAgent.__init__`](agents/base_agent.py).
