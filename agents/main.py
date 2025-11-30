# main.py - UPDATED: Integrated memory system for learning across experiments
import os, json, dotenv
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional, List, Dict, Tuple
from core.workflow import Workflow
from core.environment import Environment, Location
from core.memory_manager import WorkflowMemory

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
    experiment_id: Optional[str]
    memory_manager: Optional[WorkflowMemory]

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

def build_agentic_workflow(log_file: str = "agent_trace.txt", memory_manager: WorkflowMemory = None):
    workflow = StateGraph(AgenticState)

    planner = PlannerAgent(GEMINI_API_KEY, log_file=log_file, memory_manager=memory_manager) 
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

def run_workflow(task_description: str, state_data: dict, log_file: str = "agent_trace.txt",
                memory_manager: WorkflowMemory = None):
    """
    state_data should include:
      - env: environment parameters (as dict with DR, DE, VR, VE)
      - workflow: workflow dict (tasks, edges, N)
      - params: optional evaluator parameters (CT, CE, delta_t, delta_e)
      - experiment_id: unique identifier for this experiment
    """
    # Initialize log file with environment and workflow details
    initialize_log_file(log_file, state_data)
    
    workflow = build_agentic_workflow(log_file, memory_manager)
    
    # Add memory_manager to state so it can be used by agents
    state_data['memory_manager'] = memory_manager
    
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
    
    print(f"\n✅ Complete execution trace saved to: {log_file}")
    
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
    FIXED: Maps type-based environment parameters to all location instances.
    
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
    
    # FIX: Always ensure location 0 exists and is of type 'iot'
    if 0 not in locations_types:
        locations_types[0] = "iot"
        print("⚠️  Warning: Location 0 (IoT device) was missing. Added automatically.")
    elif locations_types[0] != "iot":
        print(f"⚠️  Warning: Location 0 was type '{locations_types[0]}'. Changed to 'iot'.")
        locations_types[0] = "iot"
    
    # ======================== PARSE ENVIRONMENT ========================
    env_data = dataset_obj['env']
    
    # Parse DR: [[0, 0, 0.0], [0, 1, 9.834e-06], ...] -> {(0, 0): 0.0, (0, 1): 9.834e-06, ...}
    # NOTE: DR in dataset is indexed by location TYPE (0, 1, 2), not location ID
    DR_type_map = {}
    for entry in env_data['DR']:
        src_type, dst_type, rate = entry
        DR_type_map[(int(src_type), int(dst_type))] = float(rate)
    
    # Parse DE/VR/VE: These are indexed by location TYPE
    DE_type_map = {int(entry[0]): float(entry[1]) for entry in env_data['DE']}
    VR_type_map = {int(entry[0]): float(entry[1]) for entry in env_data['VR']}
    VE_type_map = {int(entry[0]): float(entry[1]) for entry in env_data['VE']}
    
    # ======================== MAP TYPE-BASED PARAMS TO LOCATION IDs ========================
    # Create actual location-based maps by looking up each location's type
    
    # DE_map: {location_id: energy} based on location type
    DE_map = {}
    for loc_id, loc_type_str in locations_types.items():
        # Convert string type back to numeric for lookup
        type_num = {"iot": 0, "edge": 1, "cloud": 2}[loc_type_str]
        if type_num in DE_type_map:
            DE_map[loc_id] = DE_type_map[type_num]
        else:
            # Fallback default
            DE_map[loc_id] = 0.00012 if type_num == 0 else 2e-5
            print(f"⚠️  Warning: DE for type {type_num} missing. Using fallback.")
    
    # VR_map: {location_id: time_per_cycle} based on location type
    VR_map = {}
    for loc_id, loc_type_str in locations_types.items():
        type_num = {"iot": 0, "edge": 1, "cloud": 2}[loc_type_str]
        if type_num in VR_type_map:
            VR_map[loc_id] = VR_type_map[type_num]
        else:
            VR_map[loc_id] = 1e-7 if type_num == 0 else 2e-8
            print(f"⚠️  Warning: VR for type {type_num} missing. Using fallback.")
    
    # VE_map: {location_id: energy_per_cycle} based on location type
    VE_map = {}
    for loc_id, loc_type_str in locations_types.items():
        type_num = {"iot": 0, "edge": 1, "cloud": 2}[loc_type_str]
        if type_num in VE_type_map:
            VE_map[loc_id] = VE_type_map[type_num]
        else:
            VE_map[loc_id] = 6e-7 if type_num == 0 else 2.5e-7
            print(f"⚠️  Warning: VE for type {type_num} missing. Using fallback.")
    
    # DR_map: {(loc_i, loc_j): rate} based on location types
    # For each pair of locations, look up their types and get the DR value
    DR_map = {}
    all_location_ids = sorted(locations_types.keys())
    
    for src_id in all_location_ids:
        for dst_id in all_location_ids:
            src_type_str = locations_types[src_id]
            dst_type_str = locations_types[dst_id]
            
            src_type_num = {"iot": 0, "edge": 1, "cloud": 2}[src_type_str]
            dst_type_num = {"iot": 0, "edge": 1, "cloud": 2}[dst_type_str]
            
            if (src_type_num, dst_type_num) in DR_type_map:
                DR_map[(src_id, dst_id)] = DR_type_map[(src_type_num, dst_type_num)]
            elif src_id == dst_id:
                DR_map[(src_id, dst_id)] = 0.0  # Self-loop is always 0
            else:
                # Fallback: use a default inter-location rate
                DR_map[(src_id, dst_id)] = 1e-5
                print(f"⚠️  Warning: DR({src_type_num}, {dst_type_num}) missing. Using fallback.")
    
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

def load_dataset(json_file: str = "dataset/dataset.json") -> List[dict]:
    """Load all dataset objects from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def calculate_experiment(dataset_obj: dict, experiment_index: int, memory_manager: WorkflowMemory):
    """
    Calculate (evaluate) a single dataset object with memory integration.
    """
    experiment_id = dataset_obj.get('id', f'exp_{experiment_index}')
    
    print(f"\n{'='*80}")
    print(f"Running experiment {experiment_index} (ID: {experiment_id})")
    print(f"{'='*80}")
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
            "params": params,
            "experiment_id": experiment_id
        },
        log_file=log_file,
        memory_manager=memory_manager
    )

    # SAVE TO MEMORY
    optimal_policy = result.get("optimal_policy", [])
    evaluation_str = result.get("evaluation", "")
    plan = result.get("plan", "")
    
    # Extract evaluation result details
    evaluation_result = {
        "best_policy": optimal_policy,
        "best_cost": None,
        "evaluated": 0,
        "skipped": 0
    }
    
    # Parse evaluation string to extract metrics
    import re
    cost_match = re.search(r'U\(w,p\*\)\s*=\s*([\d.]+)', evaluation_str)
    if cost_match:
        evaluation_result["best_cost"] = float(cost_match.group(1))
    
    evaluated_match = re.search(r'Evaluated:\s*(\d+)', evaluation_str)
    if evaluated_match:
        evaluation_result["evaluated"] = int(evaluated_match.group(1))
    
    skipped_match = re.search(r'Skipped:\s*(\d+)', evaluation_str)
    if skipped_match:
        evaluation_result["skipped"] = int(skipped_match.group(1))
    
    # Save execution to memory
    memory_manager.save_execution(
        workflow_dict=workflow_dict,
        env_dict=env_dict,
        params=params,
        optimal_policy=optimal_policy,
        evaluation_result=evaluation_result,
        plan=plan,
        experiment_id=experiment_id
    )

    # DISPLAY RESULTS
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps(result.get("output", {}), indent=2))
    
    print("\n" + "="*80)
    print("OPTIMAL POLICY:")
    print("="*80)
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
    print(f"Experiment ID: {experiment_id}")
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
    # INITIALIZE MEMORY SYSTEM
    # ========================================================================
    
    print("🧠 Initializing Memory System...")
    memory_manager = WorkflowMemory(memory_dir="memory_store")
    print(f"   Memory directory: {memory_manager.memory_dir}")
    
    # ========================================================================
    # LOAD DATASET FROM JSON
    # ========================================================================
    
    print("\n📂 Loading dataset from dataset/dataset.json...")
    dataset = load_dataset("dataset/dataset.json")
    print(f"   Loaded {len(dataset)} experiment configurations\n")
    
    # Limit number of experiments for testing (set to None to run all)
    threshold = 5
    
    # Iterate over all objects and evaluate each
    for idx, dataset_obj in enumerate(dataset):
        if threshold is not None and threshold <= 0:
            break
        
        try:
            calculate_experiment(dataset_obj, idx, memory_manager)
            
            if threshold is not None:
                threshold -= 1
                
        except Exception as e:
            print(f"Error while processing experiment {idx} (ID: {dataset_obj.get('id', 'unknown')}): {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All experiments processed.")
    print(f"Memory stored in: {memory_manager.memory_dir}")
    print("="*80)