# main.py - UPDATED: Fixed Imports
import os
import json
import dotenv
import csv
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Tuple, Any

# Import new architecture components with CORRECT PATHS
from agents.orchestrator.orchestrator import AgentOrchestrator
from agents.planner_agent.planner import PlannerAgent
from agents.evaluator_agent.evaluator import EvaluatorAgent
from agents.output_agent.output import OutputAgent
from agents.base_agent.base_agent import BaseAgent

# Import core components (Ensure 'core' is in your python path)
try:
    from core.workflow import Workflow
    from core.environment import Environment
    from core.cost_eval import UtilityEvaluator
    from core.memory_manager import WorkflowMemory
except ImportError as e:
    print(f"CRITICAL WARNING: Core modules could not be imported. {e}")
    Workflow = None
    Environment = None
    UtilityEvaluator = None
    WorkflowMemory = None

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


class AgenticState(TypedDict, total=False):
    """State dictionary for the agentic workflow."""

    query: str
    env: dict
    workflow: dict
    params: Optional[dict]
    plan: Optional[str]
    evaluation: Optional[str]
    output: Optional[dict]
    optimal_policy: Optional[List[int]]
    best_cost: Optional[float]
    experiment_id: Optional[str]
    memory_manager: Optional[Any]  # Typed as Any to handle missing module


def initialize_log_file(log_file: str, state_data: dict):
    """Initialize the log file with header and environment/workflow details."""
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-AGENT TASK OFFLOADING OPTIMIZATION - EXECUTION TRACE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Execution Time: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        # Log environment details
        f.write("=" * 80 + "\n")
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        env = state_data.get("env", {})

        # Locations
        locations = env.get("locations", {})
        if locations:
            f.write("\nLocations:\n")
            f.write("-" * 40 + "\n")
            for loc_id, loc_type in sorted(locations.items()):
                f.write(f"  Location {loc_id}: {loc_type.upper()}\n")

        # Network topology (DR)
        dr = env.get("DR", {})
        if dr:
            f.write("\nNetwork Data Time Consumption (DR - ms/byte):\n")
            f.write("-" * 40 + "\n")
            for key, rate in sorted(dr.items()):
                if isinstance(key, tuple):
                    src, dst = key
                    if src != dst:  # Only show cross-location transfers
                        f.write(f"  Link ({src} → {dst}): {rate:.6e} ms/byte\n")

        # Data energy coefficients
        de = env.get("DE", {})
        if de:
            f.write("\nData Energy Consumption (DE - mJ/byte):\n")
            f.write("-" * 40 + "\n")
            for loc, coeff in sorted(de.items()):
                f.write(f"  Location {loc}: {coeff:.6e} mJ/byte\n")

        # Task time consumption
        vr = env.get("VR", {})
        if vr:
            f.write("\nTask Time Consumption (VR - ms/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, rate in sorted(vr.items()):
                f.write(f"  Location {loc}: {rate:.6e} ms/cycle\n")

        # Task energy consumption
        ve = env.get("VE", {})
        if ve:
            f.write("\nTask Energy Consumption (VE - mJ/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, energy in sorted(ve.items()):
                f.write(f"  Location {loc}: {energy:.6e} mJ/cycle\n")

        # Parameters
        params = state_data.get("params", {})
        if params:
            f.write("\nOptimization Parameters:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  CT (cost per ms): {params.get('CT', 0.2)}\n")
            f.write(f"  CE (cost per mJ): {params.get('CE', 1.34)}\n")
            f.write(f"  delta_t (time weight): {params.get('delta_t', 1)}\n")
            f.write(f"  delta_e (energy weight): {params.get('delta_e', 1)}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Log workflow details
        f.write("=" * 80 + "\n")
        f.write("WORKFLOW CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        workflow = state_data.get("workflow", {})
        tasks = workflow.get("tasks", {})
        edges_raw = workflow.get("edges", {})
        N = workflow.get("N", 0)

        # Convert edges list to dict if necessary
        edges = {}
        if isinstance(edges_raw, list):
            # Format: [{u: int, v: int, bytes: float}, ...]
            for edge in edges_raw:
                u = int(edge.get("u"))
                v = int(edge.get("v"))
                bytes_val = float(edge.get("bytes"))
                edges[(u, v)] = bytes_val
        elif isinstance(edges_raw, dict):
            edges = edges_raw

        f.write(f"\nTotal Real Tasks (N): {N}\n")
        f.write("-" * 40 + "\n")

        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            size = task_data.get("v", 0)

            f.write(f"\nTask {task_id}:\n")
            f.write(f"  CPU Cycles (v_{task_id}): {size:.2e} cycles\n")

            # Find dependencies from edges
            deps = {j: d for (i, j), d in edges.items() if i == int(task_id)}
            if deps:
                f.write(f"  Dependencies:\n")
                for dep_id, data_size in sorted(deps.items()):
                    f.write(f"    → Task {dep_id}: {data_size:.2e} bytes\n")
            else:
                f.write(f"  Dependencies: None\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("AGENT INTERACTIONS\n")
        f.write("=" * 80 + "\n\n")


def build_orchestrator(
    log_file: str = "agent_trace.txt", memory_manager=None
) -> AgentOrchestrator:
    """
    Build the agent orchestrator with all components.

    Args:
        log_file: Path to log file for agent interactions
        memory_manager: Optional memory manager for few-shot learning

    Returns:
        Configured AgentOrchestrator instance
    """
    # Create base agent for LLM access
    base_agent = BaseAgent(
        api_key=GEMINI_API_KEY, model_name="models/gemini-2.5-flash", temperature=0.3
    )

    # Create planner agent
    planner = PlannerAgent(
        api_key=GEMINI_API_KEY, log_file=log_file, memory_manager=memory_manager
    )

    # Create evaluator agent with tools
    evaluator = EvaluatorAgent(
        base_agent=base_agent,
        workflow_module=Workflow,
        environment_module=Environment,
        cost_evaluator_class=UtilityEvaluator,
        memory_manager=memory_manager,
        log_file=log_file,
    )

    # Create output agent
    output_agent = OutputAgent(api_key=GEMINI_API_KEY, log_file=log_file)

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        planner_agent=planner,
        evaluator_agent=evaluator,
        output_agent=output_agent,
        memory_manager=memory_manager,
    )

    return orchestrator


def run_workflow(
    task_description: str,
    state_data: dict,
    log_file: str = "agent_trace.txt",
    memory_manager=None,
):
    """
    Run the agentic workflow using the orchestrator.

    Args:
        task_description: Description of the task
        state_data: Initial state containing env, workflow, params
        log_file: Path to log file
        memory_manager: Optional memory manager

    Returns:
        Final state with results
    """
    # Initialize log file with environment and workflow details
    initialize_log_file(log_file, state_data)

    # Build orchestrator
    orchestrator = build_orchestrator(log_file, memory_manager)

    # Execute workflow
    print("\n" + "=" * 70)
    print("STARTING AGENTIC WORKFLOW")
    print("=" * 70)

    result = orchestrator.execute(state_data)

    # Add summary to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXECUTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Query: {task_description}\n")
        f.write(f"Optimal Policy: {result.get('optimal_policy', [])}\n")
        f.write(f"Best Cost: {result.get('best_cost', 'N/A')}\n")
        f.write(f"Evaluation: {result.get('evaluation', 'N/A')}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF TRACE\n")
        f.write("=" * 80 + "\n")

    print(f"\nComplete execution trace saved to: {log_file}")

    return result


def create_environment_dict(
    locations_types: Dict[int, str],
    DR_map: Dict[Tuple[int, int], float],
    DE_map: Dict[int, float],
    VR_map: Dict[int, float],
    VE_map: Dict[int, float],
) -> dict:
    """
    Create environment dictionary in the expected format.

    Args:
        locations_types: {location_id: type} where type in {'iot', 'edge', 'cloud'}
        DR_map: {(li, lj): ms/byte}
        DE_map: {l: mJ/byte}
        VR_map: {l: ms/cycle}
        VE_map: {l: mJ/cycle}

    Returns:
        Dictionary with locations, DR, DE, VR, VE maps
    """
    return {
        "locations": locations_types,
        "DR": DR_map,
        "DE": DE_map,
        "VR": VR_map,
        "VE": VE_map,
    }


def save_results_csv(
    out_dir: str,
    workflow_meta: dict,
    workflow_dict: dict,
    placement_policy: dict,
    optimal_cost: float,
    metrics: dict = None,
):
    """
    Save experiment results to CSV file.

    Args:
        out_dir: Output directory for CSV files
        workflow_meta: Workflow metadata
        workflow_dict: Workflow configuration
        placement_policy: Task placement mapping
        optimal_cost: Optimal cost achieved
        metrics: Additional metrics

    Returns:
        Path to created CSV file
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create filename from workflow metadata
    wf_name = workflow_meta.get("name", "unknown")
    wf_id = workflow_meta.get("id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{wf_name}_{wf_id}_{timestamp}.csv"
    csv_path = os.path.join(out_dir, csv_filename)

    # Prepare data for CSV
    rows = []
    N = workflow_dict.get("N", 0)

    # Header row
    header = ["task_id", "location", "cpu_cycles", "optimal_cost"]
    if metrics:
        header.extend(metrics.keys())

    # Data rows
    tasks = workflow_dict.get("tasks", {})
    for task_id in range(1, N + 1):
        task_data = tasks.get(task_id, {})
        location = placement_policy.get(task_id, "N/A")
        cpu_cycles = task_data.get("v", 0)

        row = {
            "task_id": task_id,
            "location": location,
            "cpu_cycles": cpu_cycles,
            "optimal_cost": optimal_cost if task_id == 1 else "",
        }

        if metrics:
            for key, value in metrics.items():
                row[key] = value if task_id == 1 else ""

        rows.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def load_dataset(dataset_path: str) -> List[dict]:
    """
    Load experiment dataset from JSON file.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        List of experiment configurations
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both array and object with "experiments" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "experiments" in data:
        return data["experiments"]
    else:
        raise ValueError(f"Unexpected dataset format in {dataset_path}")


def calculate_experiment(
    dataset_obj: dict, experiment_index: int, memory_manager: WorkflowMemory
):
    """
    Run a single experiment from the dataset.

    Args:
        dataset_obj: Experiment configuration
        experiment_index: Index of this experiment
        memory_manager: Memory manager instance
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {experiment_index}")
    print("=" * 80)

    # Extract experiment ID from meta
    meta = dataset_obj.get("meta", {})
    experiment_id = meta.get("id", f"exp_{experiment_index}")
    print(f"Experiment ID: {experiment_id}")

    # Extract workflow configuration
    workflow_dict = dataset_obj.get("workflow", {})
    N = workflow_dict.get("N", 0)
    print(f"Tasks: {N}")

    # Extract location types (convert string keys to integers)
    locations_types = dataset_obj.get("location_types", {})
    locations_types = {int(k): v for k, v in locations_types.items()}

    # Extract environment configuration
    env_data = dataset_obj.get("env", {})

    # Extract DR, DE, VR, VE matrices
    DR_map = {}
    for key, value in env_data.get("DR", {}).items():
        if isinstance(key, str):
            # Parse string key like "0,1" to tuple (removing parentheses if present)
            key_str = key.strip("()")
            parts = key_str.split(",")
            key = (int(parts[0]), int(parts[1]))
        DR_map[key] = float(value)

    DE_map = {int(k): float(v) for k, v in env_data.get("DE", {}).items()}
    VR_map = {int(k): float(v) for k, v in env_data.get("VR", {}).items()}
    VE_map = {int(k): float(v) for k, v in env_data.get("VE", {}).items()}

    # Create environment dict
    env_dict = create_environment_dict(
        locations_types=locations_types,
        DR_map=DR_map,
        DE_map=DE_map,
        VR_map=VR_map,
        VE_map=VE_map,
    )

    # Extract parameters from mode and costs
    mode = dataset_obj.get("mode", {})
    costs = dataset_obj.get("costs", {})
    params = {
        "CT": costs.get("CT", 0.2),
        "CE": costs.get("CE", 1.2),
        "delta_t": mode.get("delta_t", 1),
        "delta_e": mode.get("delta_e", 1),
    }

    print(f"Mode: ", end="")
    if params.get("delta_t") == 1 and params.get("delta_e") == 1:
        print("Balanced (Time + Energy)")
    elif params.get("delta_t") == 1 and params.get("delta_e") == 0:
        print("Low Latency (Time Only)")
    elif params.get("delta_t") == 0 and params.get("delta_e") == 1:
        print("Low Power (Energy Only)")

    # Create workflow and environment objects for validation
    wf = Workflow.from_experiment_dict(workflow_dict)
    env = Environment.from_matrices(
        types=locations_types,
        DR_matrix=DR_map,
        DE_vector=DE_map,
        VR_vector=VR_map,
        VE_vector=VE_map,
    )

    print(f"Edge Servers: {env.E}")
    print(f"Cloud Servers: {env.C}")

    # Run workflow
    log_file = f"agent_trace_exp_{experiment_index}.txt"

    result = run_workflow(
        "Find optimal offloading policy for this edge-cloud task offloading scenario",
        {
            "env": env_dict,
            "workflow": workflow_dict,
            "params": params,
            "experiment_id": experiment_id,
        },
        log_file=log_file,
        memory_manager=memory_manager,
    )

    # Save results to CSV
    try:
        # Extract policy and cost
        raw_policy = result.get("optimal_policy", [])
        best_cost = result.get("best_cost", None)

        # Convert policy list to dict
        placement_map = {}
        if isinstance(raw_policy, list):
            for i, loc in enumerate(raw_policy, start=1):
                placement_map[i] = int(loc) if loc is not None else 0
        elif isinstance(raw_policy, dict):
            placement_map = {int(k): int(v) for k, v in raw_policy.items()}

        # Prepare workflow metadata
        workflow_meta = dataset_obj.get(
            "meta", {"name": f"workflow_{experiment_index}", "id": experiment_id}
        )

        # Extract metrics from evaluation
        evaluation = result.get("evaluation", "")
        metrics = {"evaluation": evaluation}

        # Try to extract cost if not already present
        if best_cost is None:
            import re

            cost_match = re.search(r"U\(w,p\*?\)\s*=\s*([\d.]+)", str(evaluation))
            if cost_match:
                best_cost = float(cost_match.group(1))

        # Save CSV
        csv_path = save_results_csv(
            out_dir="./results_csv",
            workflow_meta=workflow_meta,
            workflow_dict=workflow_dict,
            placement_policy=placement_map,
            optimal_cost=best_cost or 0.0,
            metrics=metrics,
        )
        print(f"Saved CSV: {csv_path}")

    except Exception as e:
        print(f"Failed to save CSV for experiment {experiment_id}: {e}")

    # Save to memory
    optimal_policy = result.get("optimal_policy", [])
    evaluation_str = result.get("evaluation", "")
    plan = result.get("plan", "")

    # Extract evaluation details
    evaluation_result = {
        "best_policy": optimal_policy,
        "best_cost": result.get("best_cost"),
        "evaluated": 0,
        "skipped": 0,
    }

    # Parse evaluation string
    import re

    evaluated_match = re.search(r"Evaluated:\s*(\d+)", evaluation_str)
    if evaluated_match:
        evaluation_result["evaluated"] = int(evaluated_match.group(1))

    skipped_match = re.search(r"Skipped:\s*(\d+)", evaluation_str)
    if skipped_match:
        evaluation_result["skipped"] = int(skipped_match.group(1))

    # Save to memory
    memory_manager.save_execution(
        workflow_dict=workflow_dict,
        env_dict=env_dict,
        params=params,
        optimal_policy=optimal_policy,
        evaluation_result=evaluation_result,
        plan=plan,
        experiment_id=experiment_id,
    )

    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)

    if optimal_policy:
        print(f"Optimal Policy: {optimal_policy}")
        print(f"Best Cost: {result.get('best_cost', 'N/A')}")
        print("\nTask Assignments:")
        for i, location in enumerate(optimal_policy, start=1):
            loc_type = locations_types.get(location, "unknown")
            if location == 0:
                print(f"  Task {i} → Location {location} (IoT - Local)")
            else:
                print(f"  Task {i} → Location {location} ({loc_type.upper()})")
    else:
        print("No optimal policy found.")

    print("=" * 80)

    return result


if __name__ == "__main__":
    # ========================================================================
    # INITIALIZE MEMORY SYSTEM
    # ========================================================================

    print("Initializing Memory System...")
    memory_manager = WorkflowMemory(memory_dir="memory_store")
    print(f"   Memory directory: {memory_manager.memory_dir}")

    # ========================================================================
    # LOAD DATASET FROM JSON
    # ========================================================================
    # Limit number of experiments for testing
    start = 0
    end = 3  # Set to None to run all

    print("\nLoading dataset from dataset/dataset.json...")
    try:
        dataset = load_dataset("dataset/dataset.json")
        dataset = dataset[start:end] if end is not None else dataset[start:]
        print(f"   Loaded {len(dataset)} experiment configurations\n")
    except FileNotFoundError:
        print("   Dataset file not found. Using default example instead.\n")
        # Create a simple example if dataset doesn't exist
        dataset = [
            {
                "id": "example_1",
                "workflow": {
                    "N": 3,
                    "tasks": {1: {"v": 1e7}, 2: {"v": 5e6}, 3: {"v": 8e6}},
                    "edges": {(0, 1): 0.0, (1, 2): 2e6, (2, 3): 1.5e6, (3, 4): 0.0},
                },
                "environment": {
                    "locations": {0: "iot", 1: "edge", 2: "cloud"},
                    "DR": {
                        "(0,0)": 0.0,
                        "(0,1)": 1e-6,
                        "(0,2)": 3e-6,
                        "(1,0)": 1e-6,
                        "(1,1)": 0.0,
                        "(1,2)": 2.5e-6,
                        "(2,0)": 3e-6,
                        "(2,1)": 2.5e-6,
                        "(2,2)": 0.0,
                    },
                    "DE": {0: 1e-6, 1: 5e-7, 2: 1e-7},
                    "VR": {0: 1e-6, 1: 2e-7, 2: 1e-7},
                    "VE": {0: 1e-6, 1: 3e-7, 2: 5e-8},
                },
                "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1},
                "meta": {"name": "example", "id": "example_1"},
            }
        ]

    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================

    for idx, dataset_obj in enumerate(dataset):
        try:
            calculate_experiment(dataset_obj, idx, memory_manager)
        except Exception as e:
            print(
                f"\nError processing experiment {idx} "
                f"(ID: {dataset_obj.get('id', 'unknown')}): {e}"
            )
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print(f"   Memory stored in: {memory_manager.memory_dir}")
    print("=" * 80)
