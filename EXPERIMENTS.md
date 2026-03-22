# Running Experiments - Task Offloading Agentic System

This guide explains how to run the complete experiment pipeline for evaluating the agentic task offloading system.

## Overview

The experiment framework includes:
1. **Dataset Generation**: Creates synthetic DAG workflows + environments
2. **Baseline Comparison**: Greedy, local-only, cloud-only strategies
3. **Agentic Solution**: LLM-guided candidate generation + cost optimization
4. **Statistics**: Performance metrics (cost, time, energy, efficiency)
5. **Plotting**: Comparative visualizations

## Prerequisites

### Python Version
- Python 3.8+

### Required Packages
```bash
pip install pandas matplotlib seaborn langchain-google-genai
```

### Environment Setup
1. Obtain a **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikeys)
2. Set as environment variable:
   ```bash
   # Windows PowerShell
   $env:GOOGLE_API_KEY = "your-api-key-here"
   
   # Linux/macOS
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Running Experiments

### Full Pipeline (Recommended)
```bash
cd d:\sem 7\btp-2201cs24_2201cs31
python experiments/run_all.py
```

The script will:
1. Generate benchmark dataset (task sizes: 5, 7, 10, 15, 20 tasks)
2. Initialize agentic system (Planner + Evaluator + Output)
3. Run all experiments (baselines + agentic)
4. Compute statistics and summary
5. Generate comparison plots

**Estimated Runtime**: 10-30 minutes (depending on dataset size + API latency)

### Custom Run with API Key as Argument
```bash
python experiments/run_all.py "sk-..."
```

### Results
All outputs are saved in `experiments/results/`:
```
experiments/results/
├── benchmark_dataset.json      # Generated test cases
├── all_results.json            # Complete results
├── statistics.json             # Aggregated metrics
├── experiment_trace.log        # Detailed agent logs
└── plots/
    ├── cost_comparison.png
    ├── efficiency_curves.png
    └── ...
```

## Component Details

### Dataset Generator (`experiments/dataset.py`)
Generates synthetic workflows:
- **Task counts**: 5, 7, 10, 15, 20
- **Samples per size**: 10 (total 50 experiments)
- **Locations**: IoT, Edge, Cloud (3 locations)
- **DAG patterns**: Sequential, parallel, mixed

**Generated fields**:
- `tasks`: Task computational complexity
- `edges`: Data transfer size between tasks
- `env`: Location types + performance matrices (DR, DE, VR, VE)
- `params`: Cost weights (CT, CE, delta_t, delta_e)

### Experiment Runner (`experiments/run_experiments.py`)
Executes and compares strategies:

**Baseline Strategies**:
1. **All Local** (IoT): Tasks on device (low comm, high compute cost)
2. **All Cloud**: Tasks on cloud (high comm, low compute cost)
3. **Greedy Edge**: Offload only if beneficial (heuristic)
4. **Round-Robin**: Balance across locations

**Agentic Strategy**:
- LLM-based planning phase
- Intelligent candidate generation
- Cost-optimized placement

**Evaluation metrics**:
- Total cost `U(w,p) = delta_t*T + delta_e*E`
- Execution time `T`
- Energy consumption `E`
- Efficiency `(best_baseline_cost - agentic_cost) / best_baseline_cost`

### Experiment Plotter (`experiments/plot_results.py`)
Generates comparative visualizations:
- Cost distribution boxplots
- Efficiency improvement curves
- Task size vs. cost scatter
- Mode comparison (balanced, low-latency, low-power)

## Agent System Architecture

### Orchestrator Flow
```
AgentOrchestrator.execute(state)
  ├─ [Stage 1] PlannerAgent.run()
  │  └─ Strategic analysis + few-shot learning from memory
  ├─ [Stage 2] EvaluatorAgent.run()
  │  ├─ LLM-guided candidate generation
  │  ├─ Systematic candidate enumeration
  │  ├─ Cost evaluation + best selection
  │  └─ Optional weak solver refinement
  └─ [Stage 3] OutputAgent.run()
     └─ Result formatting + trace logging
```

### Memory System
- Location: `experiments/memory_store/`
- Purpose: Store successful workflows for few-shot learning
- Auto-enabled for all runs

### Logging
- **Agent trace**: `experiments/experiment_trace.log`
- **Detailed logs**: Recorded per experiment
- **Performance timing**: Tracked throughout pipeline

## Customizing Experiments

### Change Dataset Size
Edit `run_all.py`, line `create_dataset()`:
```python
dataset = generator.create_dataset(
    task_sizes=[5, 10, 20, 30],        # Custom task counts
    samples_per_size=20,                # More samples per size
    num_locations=4                     # 4 locations instead of 3
)
```

### Change Cost Modes
Add to baseline tests (in `run_experiments.py`):
```python
all_results[mode] = runner.run_all_experiments(
    dataset,
    run_baselines=True,
    run_agentic=True,
    delta_t=0,                          # Low-power mode
    delta_e=1
)
```

### Enable Weak Solver
Uncomment in `evaluator_agent.run()`:
```python
weak_solver.enable(algorithms=['hill_climbing', 'simulated_annealing'])
```

## Troubleshooting

### "No API key provided"
```bash
# Option 1: Environment variable
$env:GOOGLE_API_KEY = "sk-..."

# Option 2: Command line
python experiments/run_all.py "sk-..."
```

### "Missing packages"
```bash
pip install pandas matplotlib seaborn langchain-google-genai
```

### "Connection timeout"
- Check API key validity
- Verify internet connection
- Reduce dataset size (fewer samples)

### "Memory error"
- Run smaller dataset: `task_sizes=[5, 10]`
- Clear memory store: `rm -rf experiments/memory_store/`

## Understanding Results

### Statistics Output
```
EXPERIMENT SUMMARY
==================
Agentic vs Baselines (Cost - Lower is Better):
  Total Experiments:    50
  Agentic Wins:         38 (76%)
  Baseline Wins:        12 (24%)
  
  Average Cost:
    Agentic:    0.4521
    All-Cloud:  0.5890  (23% worse)
    All-Local:  0.7234  (60% worse)
    Greedy:     0.4895  (8% worse)
```

### Efficiency Metric
- **Positive**: Agentic solution better than all baselines
- **Negative**: Baseline was better (agentic still competitive)
- **Target**: > 70% wins on diverse workloads

## Output Interpretation

### Cost Breakdown (per experiment)
```json
{
  "workflow": {...},
  "optimal_policy": [0, 1, 2, 1],      // Task placements
  "best_cost": 0.3421,
  "time": 125.43,                       // ms
  "energy": 34.22,                      // mJ
  "baseline_costs": {
    "all_local": 0.5123,
    "all_cloud": 0.4891,
    "greedy": 0.3567
  }
}
```

### Efficiency Curves
- X-axis: Task count
- Y-axis: Cost/efficiency
- Shows how agentic system scales vs. baselines

## Next Steps

1. **Analyze results**: Open `experiments/results/statistics.json`
2. **Review logs**: Check `experiment_trace.log` for agent reasoning
3. **Adjust parameters**: Try different cost weights or dataset sizes
4. **Publication**: Use plots in research papers/reports

## References

- Agentic system: `agents/config.py` + orchestrator
- Core cost model: `core/cost_eval.py`
- Workflow model: `core/workflow.py`
- Environment model: `core/environment.py`
