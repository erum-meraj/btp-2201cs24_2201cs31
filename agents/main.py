# main.py - UPDATED: moved per-object evaluation into calculate_experiment() and iterate over dataset
import os, json, dotenv
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional, List, Dict, Tuple
from core.workflow import Workflow
from core.environment import Environment, Location

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

class AgenticState(TypedDict, total=False):
    query: str
    env: dict
    workflow: dict          
    params: Optional[dict]  
    plan: Optional[str]
    evaluation: Optional[str]
    output: Optional[dict]
    optimal_policy: Optional[List[int]]

def initialize_log_file(log_file: str, state_data: dict):
    """Initialize the log file with header and environment/workflow details."""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT TASK OFFLOADING OPTIMIZATION - EXECUTION TRACE\n")
        f.write("="*80 + "\n")
        f.write(f"Execution Time: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        # Log environment details
        f.write("="*80 + "\n")
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("="*80 + "\n")
        env = state_data.get('env', {})
        
        # Network topology
        dr = env.get('DR', {})
        if dr:
            f.write("\nNetwork Data Time Consumption (DR - ms/byte):\n")
            f.write("-" * 40 + "\n")
            for key, rate in sorted(dr.items()):
                if isinstance(key, tuple):
                    src, dst = key
                    f.write(f"  Link ({src} → {dst}): {rate:.6f} ms/byte\n")
        
        # Data energy coefficients
        de = env.get('DE', {})
        if de:
            f.write("\nData Energy Consumption (DE - mJ/byte):\n")
            f.write("-" * 40 + "\n")
            for loc, coeff in sorted(de.items()):
                f.write(f"  Location {loc}: {coeff:.6f} mJ/byte\n")
        
        # Task time consumption
        vr = env.get('VR', {})
        if vr:
            f.write("\nTask Time Consumption (VR - ms/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, rate in sorted(vr.items()):
                f.write(f"  Location {loc}: {rate:.6e} ms/cycle\n")
        
        # Task energy consumption
        ve = env.get('VE', {})
        if ve:
            f.write("\nTask Energy Consumption (VE - mJ/cycle):\n")
            f.write("-" * 40 + "\n")
            for loc, energy in sorted(ve.items()):
                f.write(f"  Location {loc}: {energy:.6e} mJ/cycle\n")
        
        # Parameters
        params = state_data.get('params', {})
        if params:
            f.write("\nOptimization Parameters:\n")
            f.write("-" * 40 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Log workflow details
        f.write("="*80 + "\n")
        f.write("WORKFLOW CONFIGURATION\n")
        f.write("="*80 + "\n")
        workflow = state_data.get('workflow', {})
        tasks = workflow.get('tasks', {})
        edges = workflow.get('edges', {})
        N = workflow.get('N', 0)
        
        f.write(f"\nTotal Real Tasks (N): {N}\n")
        f.write("-" * 40 + "\n")
        
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            size = task_data.get('v', 0)
            
            f.write(f"\nTask {task_id}:\n")
            f.write(f"  CPU Cycles (v_{task_id}): {size:.2e} cycles\n")
            
            # Find dependencies from edges
            deps = {j: d for (i, j), d in edges.items() if i == task_id}
            if deps:
                f.write(f"  Dependencies:\n")
                for dep_id, data_size in sorted(deps.items()):
                    f.write(f"    → Task {dep_id}: {data_size:.2e} bytes\n")
            else:
                f.write(f"  Dependencies: None\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("="*80 + "\n")
        f.write("AGENT INTERACTIONS\n")
        f.write("="*80 + "\n\n")

def build_agentic_workflow(log_file: str = "agent_trace.txt"):
    workflow = StateGraph(AgenticState)

    planner = PlannerAgent(GEMINI_API_KEY, log_file=log_file) 
    evaluator = EvaluatorAgent(GEMINI_API_KEY, log_file=log_file)
    output = OutputAgent(GEMINI_API_KEY, log_file=log_file)

    # Add nodes
    workflow.add_node("planner", planner.run)
    workflow.add_node("evaluator", evaluator.run)
    workflow.add_node("output", output.run)

    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "evaluator")
    workflow.add_edge("evaluator", "output")
    workflow.add_edge("output", END)

    return workflow.compile()

def run_workflow(task_description: str, state_data: dict, log_file: str = "agent_trace.txt"):
    """
    state_data should include:
      - env: environment parameters (as dict with DR, DE, VR, VE)
      - workflow: workflow dict (tasks, edges, N)
      - params: optional evaluator parameters (CT, CE, delta_t, delta_e)
    """
    # Initialize log file with environment and workflow details
    initialize_log_file(log_file, state_data)
    
    workflow = build_agentic_workflow(log_file)
    result = workflow.invoke({
        "query": task_description,
        **state_data  
    })

    # Add summary to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("EXECUTION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Query: {task_description}\n")
        f.write(f"Optimal Policy: {result.get('optimal_policy', [])}\n")
        f.write(f"Evaluation: {result.get('evaluation', 'N/A')}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("END OF TRACE\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Complete execution trace saved to: {log_file}")
    
    return result

def create_environment_dict(
    locations_types: Dict[int, str],
    DR_map: Dict[Tuple[int, int], float],
    DE_map: Dict[int, float],
    VR_map: Dict[int, float],
    VE_map: Dict[int, float]
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
        Dictionary with DR, DE, VR, VE maps
    """
    return {
        "locations": locations_types,
        "DR": DR_map,
        "DE": DE_map,
        "VR": VR_map,
        "VE": VE_map
    }

def parse_dataset_object(dataset_obj: dict) -> Tuple[dict, dict, dict, dict]:
    """
    Parse a dataset object from JSON and convert it to the format expected by the program.
    
    Args:
        dataset_obj: Single object from dataset.json
        
    Returns:
        Tuple of (workflow_dict, locations_types, env_dict, params)
    """
    # ======================== PARSE WORKFLOW ========================
    workflow_data = dataset_obj['workflow']
    
    # Convert tasks: {1: {"v": 20420497.0}, ...} -> {1: {"v": 20420497.0}, ...}
    tasks = {int(k): {"v": v["v"]} for k, v in workflow_data['tasks'].items()}
    
    # Convert edges: [[1, 2, 14239855.05], ...] -> {(1, 2): 14239855.05, ...}
    edges = {(int(edge[0]), int(edge[1])): edge[2] for edge in workflow_data['edges']}
    
    workflow_dict = {
        "tasks": tasks,
        "edges": edges,
        "N": workflow_data['N']
    }
    
    # ======================== PARSE LOCATION TYPES ========================
    # location_types: {1: 2, 2: 2, 3: 1, ...} where 0=iot, 1=edge, 2=cloud
    location_types_raw = {int(k): int(v) for k, v in dataset_obj['location_types'].items()}
    
    # Map numeric types to string types
    type_mapping = {0: "iot", 1: "edge", 2: "cloud"}
    locations_types = {loc: type_mapping[type_num] for loc, type_num in location_types_raw.items()}
    
    # ======================== PARSE ENVIRONMENT ========================
    env_data = dataset_obj['env']
    
    # Parse DR: [[0, 0, 0.0], [0, 1, 9.834e-06], ...] -> {(0, 0): 0.0, (0, 1): 9.834e-06, ...}
    DR_map = {}
    for entry in env_data['DR']:
        src, dst, rate = entry
        DR_map[(int(src), int(dst))] = float(rate)
    
    # Parse DE: [[0, 0.00012], [1, 2.415e-05], ...] -> {0: 0.00012, 1: 2.415e-05, ...}
    DE_map = {int(entry[0]): float(entry[1]) for entry in env_data['DE']}
    
    # Parse VR: [[0, 1e-07], [1, 1.923e-08], ...] -> {0: 1e-07, 1: 1.923e-08, ...}
    VR_map = {int(entry[0]): float(entry[1]) for entry in env_data['VR']}
    
    # Parse VE: [[0, 6e-07], [1, 2.780e-07], ...] -> {0: 6e-07, 1: 2.780e-07, ...}
    VE_map = {int(entry[0]): float(entry[1]) for entry in env_data['VE']}
    
    # Create environment dictionary
    env_dict = create_environment_dict(
        locations_types=locations_types,
        DR_map=DR_map,
        DE_map=DE_map,
        VR_map=VR_map,
        VE_map=VE_map
    )
    
    # ======================== PARSE PARAMETERS ========================
    costs = dataset_obj['costs']
    mode = dataset_obj['mode']
    
    params = {
        "CT": costs['CT'],
        "CE": costs['CE'],
        "delta_t": mode['delta_t'],
        "delta_e": mode['delta_e']
    }
    
    return workflow_dict, locations_types, env_dict, params

def load_dataset(json_file: str = "dataset.json") -> List[dict]:
    """Load all dataset objects from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def calculate_experiment(dataset_obj: dict, experiment_index: int):
    """
    Calculate (evaluate) a single dataset object. This function extracts the
    workflow/environment/params, constructs the Workflow and Environment
    objects, runs the agentic workflow, and prints the results. Uses existing
    helper functions and keeps behavior unchanged.
    """
    print(f"Running experiment {experiment_index} (ID: {dataset_obj['id']})")
    print(f"Seed: {dataset_obj['meta']['seed']}")
    print(f"Tasks: {dataset_obj['meta']['v']}, Edges: {dataset_obj['meta']['edgecount']}\n")

    # PARSE DATASET OBJECT
    workflow_dict, locations_types, env_dict, params = parse_dataset_object(dataset_obj)
    
    # Create Workflow object
    wf = Workflow.from_experiment_dict(workflow_dict)
    
    # Create Environment object
    env = Environment.from_matrices(
        types=locations_types,
        DR_matrix=env_dict["DR"],
        DE_vector=env_dict["DE"],
        VR_vector=env_dict["VR"],
        VE_vector=env_dict["VE"]
    )

    # RUN AGENTIC WORKFLOW
    log_file = f"agent_trace_exp_{experiment_index}.txt"

    result = run_workflow(
        "Find optimal offloading policy for this edge-cloud task offloading scenario", 
        {
            "env": env_dict,
            "workflow": wf.to_experiment_dict(),
            "params": params
        },
        log_file=log_file
    )

    # DISPLAY RESULTS (keeps same output format)
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps(result.get("output", {}), indent=2))
    
    print("\n" + "="*80)
    print("OPTIMAL POLICY:")
    print("="*80)
    optimal_policy = result.get("optimal_policy", [])
    if optimal_policy:
        print(f"Policy vector p = {optimal_policy}")
        print("\nTask Assignments:")
        for i, location in enumerate(optimal_policy, start=1):
            loc_type = locations_types.get(location, 'unknown')
            if location == 0:
                print(f"  Task {i} → Location {location} (IoT - Local Execution)")
            else:
                print(f"  Task {i} → Location {location} ({loc_type.capitalize()} Server)")
    else:
        print("No optimal policy found.")

    print("\n" + "="*80)
    print(f"Experiment ID: {dataset_obj['id']}")
    print(f"Number of Edge Servers (E): {env.E}")
    print(f"Number of Cloud Servers (C): {env.C}")
    print(f"Total Tasks (N): {wf.N}")
    print(f"Mode: ", end="")
    if params["delta_t"] == 1 and params["delta_e"] == 1:
        print("Balanced (Time + Energy)")
    elif params["delta_t"] == 1 and params["delta_e"] == 0:
        print("Low Latency (Time Only)")
    elif params["delta_t"] == 0 and params["delta_e"] == 1:
        print("Low Power (Energy Only)")
    print("="*80)

    return result

if __name__ == "__main__":
    # ========================================================================
    # LOAD DATASET FROM JSON
    # ========================================================================
    
    print("Loading dataset from dataset.json...")
    dataset = load_dataset("dataset.json")
    print(f"Loaded {len(dataset)} experiment configurations\n")
    threshold = 2
    # Iterate over all objects and evaluate each using the new calculate_experiment()
    for idx, dataset_obj in enumerate(dataset):
        if not threshold:
            break
        threshold -= 1
        print(f"\n{'='*80}")
        print(f"Evaluating dataset object {idx}/{len(dataset)-1}")
        print(f"{'='*80}\n")
        try:
            calculate_experiment(dataset_obj, idx)
        except Exception as e:
            print(f"Error while processing experiment {idx} (ID: {dataset_obj.get('id', 'unknown')}): {e}")

    print("\nAll experiments processed.")
