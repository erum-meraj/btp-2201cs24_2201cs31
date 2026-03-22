# Visual Guide - Complete System Architecture & Execution Flow

## 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC TASK OFFLOADING SYSTEM              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Workflow Problem                     │
│  ┌─────────────────┬──────────────────┬───────────────────────┐ │
│  │  Environment    │    Workflow DAG  │  Cost Parameters      │ │
│  │  - Locations    │  - Tasks (N)     │  - CT, CE             │ │
│  │  - DR, DE       │  - Edges         │  - delta_t/e          │ │
│  │  - VR, VE       │  - Deps          │  - Constraints        │ │
│  └─────────────────┴──────────────────┴───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AgentOrchestrator.execute()                    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ STAGE 1: PLANNER AGENT                                     ││
│  │  ├─ Strategic problem analysis                             ││
│  │  ├─ CoT reasoning with LLM                                ││
│  │  ├─ Memory-based few-shot learning                        ││
│  │  └─ Output: Strategic plan (guides evaluator)             ││
│  └────────────────────────────────────────────────────────────┘│
│                                  │
│                                  ▼
│  ┌────────────────────────────────────────────────────────────┐│
│  │ STAGE 2: EVALUATOR AGENT (Core Optimization)               ││
│  │                                                            ││
│  │  find_best_policy():                                      ││
│  │                                                            ││
│  │  ┌─────────────────────────────────────────────────────┐ ││
│  │  │ LLM Candidate Generation                            │ ││
│  │  │  └─ Generate N promising candidates from prompt    │ ││
│  │  └─────────────────────────────────────────────────────┘ ││
│  │                    │                                       ││
│  │                    ▼                                       ││
│  │  ┌─────────────────────────────────────────────────────┐ ││
│  │  │ Candidate Generator Tool                            │ ││
│  │  │  ├─ LLM candidates                                  │ ││
│  │  │  ├─ Heuristics:                                     │ ││
│  │  │  │  ├─ All-same patterns (local/edge/cloud)        │ ││
│  │  │  │  ├─ Round-robin cyclic placement                │ ││
│  │  │  │  └─ Perturbations of best so far                │ ││
│  │  │  ├─ Bounded exhaustive enumeration                 │ ││
│  │  │  └─ Deduplicate → Candidate pool (100-10k policies)│ ││
│  │  └─────────────────────────────────────────────────────┘ ││
│  │                    │                                       ││
│  │                    ▼                                       ││
│  │  ┌─────────────────────────────────────────────────────┐ ││
│  │  │ Constraint Filter Tool                              │ ││
│  │  │  ├─ Apply fixed_locations constraints              │ ││
│  │  │  ├─ Apply allowed_locations constraints            │ ││
│  │  │  └─ Output: Feasible candidates (subset)           │ ││
│  │  └─────────────────────────────────────────────────────┘ ││
│  │                    │                                       ││
│  │                    ▼                                       ││
│  │  ┌─────────────────────────────────────────────────────┐ ││
│  │  │ Utility Function Tool (Cost Evaluation)             │ ││
│  │  │                                                     │ ││
│  │  │  For each candidate policy:                        │ ││
│  │  │   1. Evaluate with UtilityEvaluator               │ ││
│  │  │   2. Cost = delta_t*T + delta_e*E                │ ││
│  │  │   3. Track: time, energy, ED, EV, delta_max      │ ││
│  │  │                                                     │ ││
│  │  │  Best selection:                                   │ ││
│  │  │   └─ Pick policy with minimum cost                │ ││
│  │  └─────────────────────────────────────────────────────┘ ││
│  │                    │                                       ││
│  │                    ▼ (optional)                            ││
│  │  ┌─────────────────────────────────────────────────────┐ ││
│  │  │ Weak Solver Tool (Future Optimization)             │ ││
│  │  │  ├─ Currently: Placeholder (no-op)                │ ││
│  │  │  ├─ Future: Local search, GA, SA, RL              │ ││
│  │  │  └─ Returns: Refined policy (if improved)         │ ││
│  │  └─────────────────────────────────────────────────────┘ ││
│  │                    │                                       ││
│  │                    ▼                                       ││
│  │  Output: best_policy, best_cost, evaluation stats        ││
│  └────────────────────────────────────────────────────────────┘│
│                                  │
│                                  ▼
│  ┌────────────────────────────────────────────────────────────┐│
│  │ STAGE 3: OUTPUT AGENT                                      ││
│  │  ├─ Format results (JSON, text, human-readable)           ││
│  │  ├─ Generate summary statistics                           ││
│  │  ├─ Log all agent reasoning and decisions                 ││
│  │  └─ Return structured final state                         ││
│  └────────────────────────────────────────────────────────────┘│
│                                  │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Optimized Solution                    │
│  ┌─────────────────┬──────────────────┬───────────────────────┐ │
│  │  Placement      │  Cost Metrics    │  Evaluation Details   │ │
│  │  - Policy vec   │  - Total cost    │  - Time breakdown     │ │
│  │  - Assignments  │  - Time (ms)     │  - Energy breakdown   │ │
│  │  - Location map │  - Energy (mJ)   │  - Agent reasoning    │ │
│  └─────────────────┴──────────────────┴───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Experiment Pipeline Flow

```
START
  │
  ├─ [1] DATASET GENERATION
  │  ├─ Task counts: [5, 7, 10, 15, 20]
  │  ├─ Samples per size: 10
  │  ├─ Locations: 3 (IoT, Edge, Cloud)
  │  └─ Output: 50 workflows → benchmark_dataset.json
  │
  ├─ [2] SYSTEM INITIALIZATION
  │  ├─ Create WorkflowMemory (for few-shot learning)
  │  ├─ Create AgentOrchestrator with:
  │  │  ├─ PlannerAgent
  │  │  ├─ EvaluatorAgent + tools
  │  │  └─ OutputAgent
  │  └─ Status: System ready
  │
  ├─ [3] EXPERIMENT EXECUTION
  │  └─ For each workflow (50 total):
  │     │
  │     ├─ RUN BASELINES:
  │     │  ├─ AllLocal: All tasks on IoT
  │     │  ├─ AllCloud: All tasks on Cloud
  │     │  ├─ Greedy: Offload if beneficial
  │     │  └─ RoundRobin: Cyclic placement
  │     │
  │     └─ RUN AGENTIC:
  │        └─ orchestrator.execute(workflow)
  │           └─ (Planner → Evaluator → Output)
  │
  │  Save results to: all_results.json
  │
  ├─ [4] STATISTICS COMPUTATION
  │  ├─ Aggregate results across all 50 experiments
  │  ├─ Compute:
  │  │  ├─ Cost improvements (agentic vs baselines)
  │  │  ├─ Win ratio (% where agentic better)
  │  │  ├─ Efficiency metrics
  │  │  └─ Confidence intervals
  │  └─ Save to: statistics.json
  │
  ├─ [5] VISUALIZATION GENERATION
  │  ├─ Cost distribution boxplots
  │  ├─ Efficiency improvement curves
  │  ├─ Task size vs. cost scatter
  │  └─ Save to: plots/
  │
  └─ END → results in experiments/results/
```

---

## 🧠 Agent Reasoning Example

### Input Problem
```
Workflow: 3 tasks (1e7, 5e6, 8e6 CPU cycles)
Edges: 1→2 (2MB), 2→3 (1.5MB)
Environment: IoT slow, Edge medium, Cloud fast
Cost: Balance time + energy (delta_t=1, delta_e=1)
```

### Agent Execution

```
┌─ PLANNER ─────────────────────────────────────────────────────┐
│                                                               │
│ Reasoning:                                                    │
│  • Task 1 is compute-heavy → benefits from fast location     │
│  • Edge → Cloud transfer expensive (2MB)                     │
│  • Local execution cheaper than remote                       │
│  • But task 1 too slow on IoT alone                         │
│                                                               │
│ Plan: "Place compute-heavy tasks on Edge, data-intensive     │
│        flows local when possible"                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌─ EVALUATOR ───────────────────────────────────────────────────┐
│                                                               │
│ LLM Candidates: [[1,1,1], [0,1,2], [0,0,0], [1,1,2]]       │
│ + Heuristics:   [[0,0,0], [2,2,2], [1,1,1], [0,1,2], ...]  │
│                                                               │
│ Candidate Pool: ~100 policies                                │
│                                                               │
│ Evaluation:                                                   │
│  [0,0,0] → Cost: 0.542 (all IoT: slow compute)             │
│  [2,2,2] → Cost: 0.489 (all Cloud: high comm)              │
│  [0,1,2] → Cost: 0.342 ✓✓✓ (BEST)                          │
│  ...                                                          │
│                                                               │
│ Result: Optimal = [0,1,2], Cost = 0.342                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌─ OUTPUT ──────────────────────────────────────────────────────┐
│                                                               │
│ Policy:        Task 1 → IoT                                 │
│                Task 2 → Edge                                │
│                Task 3 → Cloud                               │
│                                                               │
│ Metrics:       Total Cost: 0.342                            │
│                Time:       125.43 ms                        │
│                Energy:     34.22 mJ                         │
│                                                               │
│ vs Baselines:  AllLocal:  +58% worse                        │
│                AllCloud:  +43% worse                        │
│                Greedy:    +14% worse                        │
│                                                               │
│ Efficiency:    ✓ Agentic wins                               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 📊 Result Files Structure

```
experiments/results/
│
├── benchmark_dataset.json
│   └─ 50 test workflows in standardized format
│
├── all_results.json
│   └─ Per-experiment results:
│      ├─ baseline_costs: {all_local, all_cloud, greedy, rr}
│      ├─ agentic: {policy, cost, time, energy}
│      ├─ improvement: % better than best baseline
│      └─ winner: "agentic" or "baseline_name"
│
├── statistics.json
│   └─ Aggregated metrics:
│      ├─ total_experiments: 50
│      ├─ agentic_wins: 38
│      ├─ avg_improvement: 12.3%
│      ├─ by_task_count: {...}
│      └─ confidence_intervals: {...}
│
├── experiment_trace.log
│   └─ Detailed logs:
│      ├─ LLM prompts
│      ├─ Agent reasoning
│      ├─ Candidate generation
│      └─ Cost evaluations
│
└── plots/
    ├── cost_comparison.png       # Boxplot baseline vs agentic
    ├── efficiency_curve.png      # Improvement over task count
    ├── scatter_cost_vs_tasks.png # Task complexity analysis
    └── mode_comparison.png       # Balanced/Low-latency/Low-power
```

---

## 🔑 Key Metrics Explained

### Cost (Lower is Better)
$$U(w,p) = \delta_t \cdot T + \delta_e \cdot E$$

- `T`: Execution time (ms)
- `E`: Energy consumption (mJ)
- `delta_t`, `delta_e`: Weights (1 = balanced, 0 = ignore)

### Efficiency
$$\text{Efficiency} = \frac{\text{Best\_Baseline} - \text{Agentic}}{\text{Best\_Baseline}} \times 100\%$$

- Positive = Agentic wins
- Negative = Baseline better (agentic still valid)

### Win Ratio
$$\text{WinRatio} = \frac{\text{Agentic Wins}}{\text{Total Tests}} \times 100\%$$

- Target: > 70% for practical system
- > 90% = Excellent optimization

---

## 🚀 Running This Visually

### Step 1: Quick Test
```bash
python quickstart.py
     │
     ├─ Load small problem (3 tasks)
     ├─ Run orchestrator
     ├─ Show optimal placement
     └─ Time: ~2-5 min
```

### Step 2: Full Benchmark
```bash
python experiments/run_all.py
     │
     ├─ [████████░░] Generate dataset (1 min)
     ├─ [████████░░] Initialize system (1 min)
     ├─ [████████░░] Run 50 experiments (20 min)
     ├─ [████████░░] Compute statistics (2 min)
     ├─ [████████░░] Generate plots (2 min)
     └─ Done: Check experiments/results/
```

---

## ✅ Verification Checklist

- [ ] API key set (`$env:GOOGLE_API_KEY`)
- [ ] Python packages installed (`pandas`, `matplotlib`, `seaborn`)
- [ ] `quickstart.py` runs successfully
- [ ] Results appear in `experiments/results/`
- [ ] Statistics show agentic > 50% wins
- [ ] Plots generated without errors
- [ ] Trace log captured agent reasoning

**If all pass: ✅ System working correctly!**
