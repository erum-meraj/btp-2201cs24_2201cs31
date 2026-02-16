# Agentic Task Offloading System

> A multi-agent system for optimizing task placement in edge-cloud computing environments using LLM-guided search and mathematical evaluation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Understanding the Evaluation Flow](#understanding-the-evaluation-flow)
- [Optimization Modes](#optimization-modes)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Contributing](#contributing)
- [Research Paper](#research-paper)
- [License](#license)

---

## 🎯 Overview

The **Agentic Task Offloading System** is a sophisticated multi-agent framework designed to solve the task offloading optimization problem in edge-cloud computing environments. It combines:

- **Large Language Model (LLM)** reasoning for intelligent heuristics
- **Mathematical optimization** for exact cost computation
- **Memory-based learning** from past executions
- **Adaptive search strategies** that scale from small to large problems

### The Problem

In edge-cloud computing, IoT devices must decide where to execute their tasks:

- **Locally** (IoT device): Low latency, high energy
- **Edge servers**: Balanced performance
- **Cloud servers**: High compute power, higher latency

The challenge: Find the optimal placement policy that minimizes total cost (time + energy).

### The Solution

This system uses a **three-agent architecture** to find optimal task placements:

1. **Planner Agent**: Analyzes the problem and creates a strategic plan
2. **Evaluator Agent**: Generates and evaluates candidate policies using mathematical models
3. **Output Agent**: Formats results with clear explanations

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture

- **Modular design** with specialized agents
- **Clean separation** between strategic planning and evaluation
- **Tool-based evaluator** with pluggable components

### 🧠 Intelligent Search

- **LLM-guided heuristics** for smart candidate generation
- **Memory-based learning** from similar past executions
- **Adaptive exhaustive search** for small problems
- **Systematic patterns** for comprehensive coverage

### 📐 Mathematical Precision

- **Exact cost computation** using research paper equations
- **Critical path analysis** for delay-DAG
- **Energy and time modeling** based on hardware parameters
- **Provably optimal** solutions for small problems

### 🎯 Flexible Optimization

- **Three optimization modes**: Balanced, Low Latency, Low Power
- **Constraint support**: Fixed locations, allowed locations
- **Multi-objective**: Time and energy trade-offs

### 💾 Memory System

- **Stores execution history** for learning
- **Retrieves similar cases** for few-shot learning
- **Improves over time** as more problems are solved

### 📊 Comprehensive Logging

- **Complete execution traces** with timestamps
- **Agent interactions** logged for debugging
- **Cost breakdowns** with detailed metrics

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                             │
│                                                               │
│  • Coordinates workflow execution                             │
│  • Manages state transitions                                  │
│  • Integrates memory system                                   │
└───────────────────┬──────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ PLANNER  │→ │EVALUATOR │→ │  OUTPUT  │
│  AGENT   │  │  AGENT   │  │  AGENT   │
└──────────┘  └────┬─────┘  └──────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Candidate│ │Utility │ │ Weak  │
    │Generator│ │  Tool  │ │Solver │
    └────────┘ └────────┘ └────────┘
```

### Agent Responsibilities

#### 1. Planner Agent

**Role:** Strategic Analysis

- Analyzes environment characteristics (bandwidth, compute power)
- Examines workflow structure (task dependencies, data sizes)
- Considers optimization mode (time vs energy)
- Generates strategic guidance for the evaluator

**Output:** Natural language strategic plan

#### 2. Evaluator Agent

**Role:** Policy Search & Evaluation (Supervisor)

- Coordinates multiple search strategies
- Generates candidate policies using:
  - LLM-guided intelligent suggestions
  - Memory-based similar executions
  - Systematic heuristics
  - Exhaustive enumeration (if feasible)
- Evaluates each candidate using mathematical cost model
- Selects policy with minimum cost

**Output:** Optimal policy vector and cost

#### 3. Output Agent

**Role:** Result Presentation

- Formats the optimal policy
- Creates task-to-location mappings
- Provides cost breakdowns
- Generates natural language explanations

**Output:** User-friendly formatted results

---

## 🔄 How It Works

### The Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Problem Definition                                    │
├─────────────────────────────────────────────────────────────┤
│ • Environment: DR, DE, VR, VE (network & compute params)    │
│ • Workflow: Tasks, dependencies (DAG structure)             │
│ • Parameters: CT, CE, delta_t, delta_e (cost weights)       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: STRATEGIC PLANNING (LLM)                           │
├─────────────────────────────────────────────────────────────┤
│ Planner Agent analyzes:                                      │
│ • "Task 1 is compute-intensive → prefer cloud"              │
│ • "Tasks 2-3 have high data dependency → co-locate"         │
│ • "Network bandwidth to edge is limited"                    │
│                                                              │
│ Produces: Strategic plan (text)                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: CANDIDATE GENERATION (LLM + Logic)                 │
├─────────────────────────────────────────────────────────────┤
│ Step 2.1: LLM Suggestions                                   │
│   LLM: "Try [2,1,1,2], [0,2,2,2], [2,2,1,2]"               │
│                                                              │
│ Step 2.2: Memory Retrieval                                  │
│   Memory: "Similar workflow used [1,1,2,2]"                │
│                                                              │
│ Step 2.3: Systematic Patterns                               │
│   Logic: "[0,0,0,0], [1,1,1,1], [2,2,2,2]"                │
│                                                              │
│ Step 2.4: Exhaustive (if N^L < 10,000)                     │
│   Logic: "Generate all 81 combinations"                     │
│                                                              │
│ Result: ~81 candidate policies                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MATHEMATICAL EVALUATION (Pure Math, No LLM)       │
├─────────────────────────────────────────────────────────────┤
│ For each of 81 candidates:                                  │
│                                                              │
│   policy = [2,1,1,2]                                        │
│                                                              │
│   1. Compute Data Energy (Equation 4):                      │
│      ED = Σ DE(li) × (incoming + outgoing data)            │
│                                                              │
│   2. Compute Task Energy (Equation 5):                      │
│      EV = Σ vi × VE(li)                                     │
│                                                              │
│   3. Compute Total Energy (Equation 3):                     │
│      E = CE × (ED + EV)                                     │
│                                                              │
│   4. Compute Critical Path (Equation 6):                    │
│      Build delay-DAG, find longest path                     │
│                                                              │
│   5. Compute Time Cost (Equation 7):                        │
│      T = CT × Δ_max                                         │
│                                                              │
│   6. Compute Total Cost (Equation 8):                       │
│      U(w,p) = δ_t × T + δ_e × E                            │
│                                                              │
│   Track best: min(U(w,p))                                   │
│                                                              │
│ Result: Policy [2,1,1,2] with cost 67.123                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: OUTPUT FORMATTING (LLM)                            │
├─────────────────────────────────────────────────────────────┤
│ Output Agent creates:                                        │
│ • Task mappings: "Task 1 → Cloud (location 2)"             │
│ • Cost breakdown: "Time: 45.2ms, Energy: 21.9mJ"           │
│ • Explanation: "Heavy tasks placed on cloud for speed..."   │
│                                                              │
│ Result: Formatted output with explanations                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: LLM vs Math

| Component         | LLM Role                | Math Role              |
| ----------------- | ----------------------- | ---------------------- |
| **Planner**       | ✅ Analyze & strategize | ❌                     |
| **Candidate Gen** | ✅ Suggest policies     | ❌                     |
| **Evaluation**    | ❌                      | ✅ Compute exact costs |
| **Output**        | ✅ Explain results      | ❌                     |

**The LLM never computes costs—it only suggests candidates. Mathematics does the actual evaluation using research paper equations.**

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (for LLM)
- 4GB+ RAM recommended

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/agentic-task-offloading.git
cd agentic-task-offloading
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

```
langchain-google-genai>=0.1.0
python-dotenv>=1.0.0
numpy>=1.21.0
```

### Step 3: Set Up API Key

Create a `.env` file:

```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

Or set environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Step 4: Verify Installation

```bash
python test_architecture.py
```

Expected output:

```
✓ All tests passed! Architecture is valid.
```

---

## 🚀 Quick Start

### Example 1: Simple 3-Task Problem

```python
from main import run_workflow, create_environment_dict
from core.memory_manager import WorkflowMemory

# Initialize memory system
memory = WorkflowMemory(memory_dir="memory_store")

# Define environment
env = create_environment_dict(
    locations_types={0: 'iot', 1: 'edge', 2: 'cloud'},
    DR_map={
        (0, 0): 0.0, (0, 1): 1e-6, (0, 2): 3e-6,
        (1, 0): 1e-6, (1, 1): 0.0, (1, 2): 2e-6,
        (2, 0): 3e-6, (2, 1): 2e-6, (2, 2): 0.0
    },
    DE_map={0: 1e-6, 1: 5e-7, 2: 1e-7},
    VR_map={0: 1e-6, 1: 2e-7, 2: 1e-7},
    VE_map={0: 1e-6, 1: 3e-7, 2: 5e-8}
)

# Define workflow
workflow = {
    'N': 3,
    'tasks': {
        1: {'v': 1e7},   # 10M CPU cycles
        2: {'v': 5e6},   # 5M CPU cycles
        3: {'v': 8e6}    # 8M CPU cycles
    },
    'edges': {
        (0, 1): 0.0,      # Entry → Task 1
        (1, 2): 2e6,      # Task 1 → Task 2 (2MB data)
        (2, 3): 1.5e6,    # Task 2 → Task 3 (1.5MB data)
        (3, 4): 0.0       # Task 3 → Exit
    }
}

# Define parameters (Balanced mode)
params = {
    'CT': 0.2,      # Cost per millisecond
    'CE': 1.34,     # Cost per millijoule
    'delta_t': 1,   # Time weight
    'delta_e': 1    # Energy weight
}

# Run optimization
result = run_workflow(
    task_description="Find optimal offloading policy",
    state_data={
        'env': env,
        'workflow': workflow,
        'params': params,
        'experiment_id': 'example_1'
    },
    log_file="example_trace.txt",
    memory_manager=memory
)

# View results
print(f"Optimal Policy: {result['optimal_policy']}")
print(f"Best Cost: {result['best_cost']}")
```

**Output:**

```
Optimal Policy: [2, 1, 2]
Best Cost: 45.678

Task Assignments:
  Task 1 → Location 2 (Cloud)
  Task 2 → Location 1 (Edge)
  Task 3 → Location 2 (Cloud)
```

### Example 2: Using the Orchestrator Directly

```python
from config import create_system

# Create system (reads API key from environment)
orchestrator = create_system(memory_manager=memory)

# Run optimization
result = orchestrator.execute({
    'env': env,
    'workflow': workflow,
    'params': params
})
```

### Example 3: Batch Processing with Dataset

```bash
# Process multiple experiments from JSON dataset
python main.py
```

This will:

1. Load experiments from `dataset/dataset.json`
2. Process each experiment sequentially
3. Save results to CSV files in `results_csv/`
4. Store executions in memory for learning
5. Generate trace logs for each experiment

---

## 📖 Detailed Usage

### Defining Environments

An environment specifies the edge-cloud network topology:

```python
env = {
    'locations': {
        0: 'iot',      # Always present (IoT device)
        1: 'edge',     # Edge server 1
        2: 'edge',     # Edge server 2
        3: 'cloud'     # Cloud server
    },

    # Data Transfer Time (ms/byte)
    'DR': {
        (0, 1): 1e-6,    # IoT → Edge1: 1 microsecond per byte
        (0, 3): 3e-6,    # IoT → Cloud: 3 microseconds per byte
        # ... all pairs
    },

    # Data Energy (mJ/byte)
    'DE': {
        0: 1e-6,   # IoT uses 1 microjoule per byte
        1: 5e-7,   # Edge uses 0.5 microjoules per byte
        3: 1e-7    # Cloud uses 0.1 microjoules per byte
    },

    # Task Execution Time (ms/cycle)
    'VR': {
        0: 1e-6,   # IoT: 1 microsecond per CPU cycle
        1: 2e-7,   # Edge: 0.2 microseconds per cycle (5x faster)
        3: 1e-7    # Cloud: 0.1 microseconds per cycle (10x faster)
    },

    # Task Energy (mJ/cycle)
    'VE': {
        0: 1e-6,   # IoT: 1 microjoule per cycle
        1: 3e-7,   # Edge: 0.3 microjoules per cycle
        3: 5e-8    # Cloud: 0.05 microjoules per cycle
    }
}
```

### Defining Workflows

A workflow represents the application's task structure as a DAG:

```python
workflow = {
    'N': 4,  # Number of real tasks (excluding entry/exit)

    'tasks': {
        1: {'v': 1e7},    # Task 1: 10M CPU cycles
        2: {'v': 5e6},    # Task 2: 5M CPU cycles
        3: {'v': 8e6},    # Task 3: 8M CPU cycles
        4: {'v': 1.5e7}   # Task 4: 15M CPU cycles
    },

    'edges': {
        # (source, target): data_size_in_bytes
        (0, 1): 0.0,        # Entry → Task 1 (no data)
        (1, 2): 2e6,        # Task 1 → Task 2 (2MB)
        (1, 3): 1.5e6,      # Task 1 → Task 3 (1.5MB)
        (2, 4): 3e6,        # Task 2 → Task 4 (3MB)
        (3, 4): 2.5e6,      # Task 3 → Task 4 (2.5MB)
        (4, 5): 0.0         # Task 4 → Exit (no data)
    }
}
```

**Workflow Visualization:**

```
       Task 1 (10M cycles)
      /              \
   2MB              1.5MB
    /                  \
Task 2 (5M)        Task 3 (8M)
    \                  /
   3MB              2.5MB
      \              /
       Task 4 (15M cycles)
```

### Setting Parameters

```python
# Balanced Mode: Optimize both time and energy equally
params = {
    'CT': 0.2,      # Cost per unit time (default: 1/5ms = 0.2)
    'CE': 1.34,     # Cost per unit energy (default: 1/0.746mJ ≈ 1.34)
    'delta_t': 1,   # Time weight (0 or 1)
    'delta_e': 1    # Energy weight (0 or 1)
}

# Low Latency Mode: Minimize time only
params = {
    'CT': 0.2,
    'CE': 1.34,
    'delta_t': 1,   # Consider time
    'delta_e': 0    # Ignore energy
}

# Low Power Mode: Minimize energy only
params = {
    'CT': 0.2,
    'CE': 1.34,
    'delta_t': 0,   # Ignore time
    'delta_e': 1    # Consider energy
}
```

### Adding Constraints

```python
params = {
    'CT': 0.2,
    'CE': 1.34,
    'delta_t': 1,
    'delta_e': 1,

    # Task 1 MUST run locally
    'fixed_locations': {
        1: 0  # Task 1 → Location 0 (IoT)
    },

    # Task 2 can ONLY run on edge servers
    'allowed_locations': {
        2: [1, 2]  # Task 2 → Locations 1 or 2 only
    }
}
```

---

## ⚙️ Configuration

### Using Environment Variables

```bash
# Required
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional
export MODEL_NAME="models/gemini-2.0-flash"
export TEMPERATURE="0.3"
```

### Programmatic Configuration

```python
from config import AgentConfig, SystemBuilder

# Custom configuration
config = AgentConfig(
    api_key="your-api-key",
    log_file="custom_trace.txt",
    model_name="models/gemini-2.0-flash",
    temperature=0.3
)

# Build system
builder = SystemBuilder(config)
orchestrator = builder.build_orchestrator(memory_manager=memory)
```

### Adjusting Search Parameters

```python
from evaluator_agent import EvaluatorAgent

# Create evaluator with custom exhaustive search limit
evaluator = EvaluatorAgent(...)

# Default: 10,000 combinations
# Increase for more aggressive exhaustive search
evaluator.candidate_generator.set_exhaustive_limit(50000)

# Decrease for more conservative (faster) search
evaluator.candidate_generator.set_exhaustive_limit(5000)

# Disable exhaustive search entirely
evaluator.candidate_generator.set_exhaustive_limit(0)
```

---

## 🔍 Understanding the Evaluation Flow

See [EVALUATION_FLOW.md](EVALUATION_FLOW.md) for a complete, detailed explanation of how policies are evaluated.

### Quick Summary

1. **Planner Agent** (LLM): Creates strategic plan
2. **Evaluator Agent**:
   - **Step 2.1** (LLM): Suggests 3-5 intelligent candidate policies
   - **Step 2.2** (Logic): Generates systematic patterns
   - **Step 2.3** (Logic): Adds all combinations if problem is small
   - **Step 2.4** (MATH): Evaluates each candidate using equations
3. **Output Agent** (LLM): Formats and explains results

**Key:** LLM suggests candidates, mathematics evaluates them exactly.

---

## 🎚️ Optimization Modes

### Balanced Mode (Default)

**Use when:** You want to optimize both time and energy

```python
params = {'CT': 0.2, 'CE': 1.34, 'delta_t': 1, 'delta_e': 1}
```

**Example:** Mobile applications that need good performance but also care about battery life.

### Low Latency Mode

**Use when:** Time is critical, energy doesn't matter

```python
params = {'CT': 0.2, 'CE': 1.34, 'delta_t': 1, 'delta_e': 0}
```

**Example:** Real-time video processing, gaming, emergency response systems.

### Low Power Mode

**Use when:** Battery life is critical, some delay is acceptable

```python
params = {'CT': 0.2, 'CE': 1.34, 'delta_t': 0, 'delta_e': 1}
```

**Example:** Sensor networks, wearable devices, battery-constrained IoT.

---

## 🚀 Advanced Features

### Memory-Based Learning

The system learns from past executions:

```python
# Initialize memory
memory = WorkflowMemory(memory_dir="memory_store")

# Use in orchestrator
orchestrator = create_system(memory_manager=memory)

# Run multiple experiments
for experiment in experiments:
    result = orchestrator.execute(experiment)
    # Memory automatically stores results

# Future runs will use past executions for better suggestions
```

**Benefits:**

- Faster convergence on similar problems
- Better initial candidates
- Improves with experience

### Weak Solver (Placeholder)

For future advanced optimization:

```python
# Enable weak solver (currently placeholder)
evaluator.weak_solver.enable(['genetic_algorithm'])

# Future implementations will include:
# - Genetic algorithms
# - Simulated annealing
# - Reinforcement learning
```

### Custom Tools

Extend the evaluator with custom optimization:

```python
class MyCustomOptimizer:
    def optimize(self, candidates, workflow, env):
        # Your custom optimization logic
        return improved_policy

# Add to evaluator
evaluator.custom_optimizer = MyCustomOptimizer()
```

---

## 📁 Project Structure

```
agentic-task-offloading/
│
├── main.py                          # Main entry point
├── orchestrator.py                  # Workflow coordinator
├── config.py                        # Configuration utilities
│
├── agents/                          # Agent implementations
│   ├── planner.py                   # Strategic planning agent
│   ├── output.py                    # Output formatting agent
│   └── base_agent/
│       └── base_agent.py            # Base LLM agent class
│
├── evaluator_agent.py               # Policy evaluation supervisor
├── candidate_generator.py           # Policy generation tool
├── utility_tool.py                  # Cost evaluation tool
├── weak_solver.py                   # Advanced optimization (placeholder)
│
├── core/                            # Core domain models
│   ├── workflow.py                  # Workflow DAG model
│   ├── environment.py               # Environment model
│   ├── cost_eval.py                 # Cost evaluation (equations 1-8)
│   ├── memory_manager.py            # Execution memory
│   └── utils.py                     # Utility functions
│
├── dataset/                         # Experiment datasets
│   └── dataset.json                 # Problem configurations
│
├── results_csv/                     # Output CSV files
│   └── *.csv                        # Experiment results
│
├── memory_store/                    # Execution history
│   └── *.json                       # Stored executions
│
├── tests/                           # Test suite
│   └── test_architecture.py         # Architecture tests
│
├── docs/                            # Documentation
│   ├── README.md                    # This file
│   ├── QUICKSTART.md                # Quick start guide
│   ├── ARCHITECTURE.md              # System architecture
│   ├── EVALUATION_FLOW.md           # Evaluation flow details
│   ├── MIGRATION.md                 # Migration guide
│   └── API.md                       # API reference
│
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables
└── LICENSE                          # License file
```

---

## 📚 API Reference

### Orchestrator

```python
from orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    planner_agent,      # PlannerAgent instance
    evaluator_agent,    # EvaluatorAgent instance
    output_agent,       # OutputAgent instance
    memory_manager      # Optional WorkflowMemory instance
)

result = orchestrator.execute(state)
# Returns: dict with 'plan', 'evaluation', 'optimal_policy', 'best_cost', 'output'
```

### Planner Agent

```python
from agents.planner import PlannerAgent

planner = PlannerAgent(
    api_key,            # Google Gemini API key
    log_file,           # Path to log file
    memory_manager      # Optional memory manager
)

result = planner.run(state)
# Returns: state with 'plan' added
```

### Evaluator Agent

```python
from evaluator_agent import EvaluatorAgent

evaluator = EvaluatorAgent(
    base_agent,                 # BaseAgent for LLM access
    workflow_module,            # Workflow class
    environment_module,         # Environment class
    cost_evaluator_class,       # UtilityEvaluator class
    memory_manager,             # Optional memory manager
    log_file                    # Path to log file
)

result = evaluator.run(state)
# Returns: state with 'evaluation', 'optimal_policy', 'best_cost' added
```

### Candidate Generator

```python
from candidate_generator import CandidatePolicyGenerator

generator = CandidatePolicyGenerator(
    memory_manager,        # Optional memory manager
    max_exhaustive=10000   # Max combinations for exhaustive search
)

candidates = generator.generate_candidates(
    num_tasks,         # Number of tasks
    location_ids,      # List of location IDs
    workflow_dict,     # Workflow configuration
    env_dict,          # Environment configuration
    params,            # Parameters
    llm_candidates     # Optional LLM suggestions
)
# Returns: List of policy tuples
```

### Utility Tool

```python
from utility_tool import UtilityFunctionTool

tool = UtilityFunctionTool(evaluator)  # UtilityEvaluator instance

result = tool.find_best_policy(
    candidates,    # List of policy tuples
    workflow,      # Workflow object
    environment    # Environment object
)
# Returns: dict with 'best_policy', 'best_cost', 'evaluated', 'skipped'
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error:**

```
ValueError: API key not found
```

**Solution:**

```bash
# Set environment variable
export GOOGLE_API_KEY="your-key"

# Or create .env file
echo "GOOGLE_API_KEY=your-key" > .env
```

#### 2. Import Errors

**Error:**

```
ModuleNotFoundError: No module named 'orchestrator'
```

**Solution:**

```python
import sys
sys.path.append('/path/to/project/root')
```

#### 3. All Policies Have Infinite Cost

**Problem:** No valid policies found

**Causes:**

- Invalid environment parameters
- Disconnected locations in network
- Unrealistic DR/VR values

**Solution:**

```python
# Verify environment connectivity
for i in location_ids:
    for j in location_ids:
        assert (i, j) in DR_map, f"Missing DR({i},{j})"

# Check for reasonable values
assert all(dr > 0 for dr in DR_map.values()), "DR must be positive"
```

#### 4. Out of Memory

**Problem:** System crashes during exhaustive search

**Solution:**

```python
# Reduce exhaustive search limit
evaluator.candidate_generator.set_exhaustive_limit(5000)

# Or disable entirely
evaluator.candidate_generator.set_exhaustive_limit(0)
```

#### 5. Slow Performance

**Problem:** Takes too long to find solution

**Causes:**

- Large problem (many tasks/locations)
- Exhaustive search on large space

**Solution:**

```python
# Check problem size
num_combos = len(location_ids) ** num_tasks
print(f"Problem size: {num_combos:,} combinations")

# If > 10,000, exhaustive is automatically disabled
# If still slow, reduce LLM candidates or heuristics
```

---

## ⚡ Performance Tuning

### Problem Size Guidelines

| Tasks | Locations | Combinations | Expected Time | Recommendation  |
| ----- | --------- | ------------ | ------------- | --------------- |
| 3     | 3         | 27           | < 1 sec       | Use exhaustive  |
| 4     | 4         | 256          | < 5 sec       | Use exhaustive  |
| 5     | 5         | 3,125        | < 30 sec      | Use exhaustive  |
| 7     | 4         | 16,384       | 2-5 min       | Heuristics only |
| 10    | 5         | 9.7M         | Hours         | Heuristics only |

### Optimization Strategies

#### For Small Problems (< 10K combinations)

```python
# Use default settings (exhaustive search)
max_exhaustive = 10000  # Default
```

#### For Medium Problems (10K - 100K)

```python
# Disable exhaustive, rely on heuristics
max_exhaustive = 0

# Increase LLM suggestions
# Modify evaluator prompt to request more candidates
```

#### For Large Problems (> 100K)

```python
# Minimal candidate set
max_exhaustive = 0

# Consider implementing weak solver
evaluator.weak_solver.enable(['genetic_algorithm'])

# Use memory aggressively
memory_manager.retrieve_similar_executions(top_k=10)
```

### Memory Management

```python
# Limit memory size
memory = WorkflowMemory(
    memory_dir="memory_store",
    max_executions=1000  # Keep only 1000 most recent
)

# Clear old executions
memory.clear_old_executions(days=30)
```

### Logging Performance

```python
# Disable verbose logging for production
import logging
logging.getLogger().setLevel(logging.WARNING)

# Disable progress updates
show_progress = False
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/agentic-task-offloading.git
cd agentic-task-offloading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run architecture verification
python test_architecture.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_architecture.py

# Run with coverage
pytest --cov=. tests/
```

---

## 📄 Research Paper

This implementation is based on:

**"Deep Meta Q-Learning Based Multi-Task Offloading in Edge-Cloud Systems"**

- Authors: Nelson Sharma, Aswini Ghosh, Rajiv Misra, Sajal K. Das
- Published in: IEEE Transactions on Mobile Computing, Vol. 23, No. 4, April 2024
- DOI: 10.1109/TMC.2023.3264901

### Cost Model (Equations 1-8)

The system implements the complete cost model from the paper:

```
U(w,p) = δ_t × T + δ_e × E                    (Equation 8)

where:
  T = CT × Δ_max(delay-DAG)                    (Equation 7)
  E = CE × (ED + EV)                           (Equation 3)

  ED = Σ DE(l_i) × (Σ d_{j,i} + Σ d_{i,k})   (Equation 4)
  EV = Σ v_i × VE(l_i)                         (Equation 5)

  D_Δ(i,j) = d_{i,j} × DR(l_i,l_j) + v_i × VR(l_i)  (Equation 6)
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Based on research by Sharma et al. (IEEE TMC 2024)
- Uses Google Gemini for LLM capabilities
- Built with LangChain framework

---

## 📞 Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/your-org/agentic-task-offloading/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-org/agentic-task-offloading/discussions)

---

## 🗺️ Roadmap

### Current Version (v1.0)

- ✅ Multi-agent architecture
- ✅ LLM-guided search
- ✅ Mathematical evaluation
- ✅ Memory system
- ✅ Three optimization modes

### Upcoming (v1.1)

- 🔄 Weak solver implementation
- 🔄 Parallel policy evaluation
- 🔄 Web UI dashboard
- 🔄 Real-time monitoring

### Future (v2.0)

- 📋 Multi-objective optimization
- 📋 Dynamic environment adaptation
- 📋 Federated learning support
- 📋 GPU acceleration

---

## 📊 Citation

If you use this system in your research, please cite:

```bibtex
@article{sharma2024deep,
  title={Deep Meta Q-Learning Based Multi-Task Offloading in Edge-Cloud Systems},
  author={Sharma, Nelson and Ghosh, Aswini and Misra, Rajiv and Das, Sajal K},
  journal={IEEE Transactions on Mobile Computing},
  volume={23},
  number={4},
  pages={2583--2598},
  year={2024},
  publisher={IEEE}
}
```

---

**Made with ❤️ for edge-cloud computing research**
