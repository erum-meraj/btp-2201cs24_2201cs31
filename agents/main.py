# main.py - UPDATED with correct experiment initialization
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

if __name__ == "__main__":
    # ========================================================================
    # EXPERIMENT SETUP: Define workflow, environment, and optimization params
    # ========================================================================
    
    # ------------------------ WORKFLOW DEFINITION ---------------------------
    # Define workflow as per the paper's format
    workflow_dict = {
    "tasks": {
        1: {"v": 2e6},    # light
        2: {"v": 10e6},   # medium
        3: {"v": 35e6},   # heavy
        4: {"v": 18e6},   # medium
        5: {"v": 28e6},   # heavy
        6: {"v": 6e6},    # light
    },
    "edges": {
        (1, 2): 15e6,     # 15 MB (big)
        (2, 3): 0.8e6,    # 0.8 MB
        (3, 4): 2e6,      # 2 MB
        (4, 5): 1e6,      # 1 MB
        (5, 6): 0.6e6,    # 0.6 MB
    },
    "N": 6,
}

    
    # Create Workflow object from experiment dict
    wf = Workflow.from_experiment_dict(workflow_dict)
    
    # ------------------------ ENVIRONMENT DEFINITION ------------------------
    # Define location types: 0=IoT (mandatory), 1+=edge/cloud
    locations_types = {0: "iot", 1: "edge_a", 2: "edge_b", 3: "cloud"}
    
    # DR: Data Time Consumption (ms/byte) - time to transfer 1 byte between locations
    DR_map = {
    (0,0):0.0, (1,1):0.0, (2,2):0.0, (3,3):0.0,
    (0,1):1.0e-05, (1,0):1.0e-05,   # IoT <-> Edge-A: 10 ms/MB
    (0,2):1.5e-05, (2,0):1.5e-05,   # IoT <-> Edge-B: 15 ms/MB
    (0,3):2.0e-03, (3,0):2.0e-03,   # IoT <-> Cloud: 2000 ms/MB (slow)
    (1,2):4.0e-05, (2,1):4.0e-05,   # Edge-A <-> Edge-B: 40 ms/MB
    (1,3):6.0e-05, (3,1):6.0e-05,   # Edge-A <-> Cloud: 60 ms/MB
    (2,3):3.0e-05, (3,2):3.0e-05,   # Edge-B <-> Cloud: 30 ms/MB (fastest edge↔cloud)
}


    
    # DE: Data Energy Consumption (mJ/byte) - energy to process 1 byte at location
    DE_map = {0: 1.20e-4, 1: 2.50e-5, 2: 2.20e-5, 3: 1.80e-5}
    
    # VR: Task Time Consumption (ms/cycle) - time to execute 1 CPU cycle
    VR_map = {0: 1.0e-7, 1: 3.0e-8, 2: 2.0e-8, 3: 1.0e-8}
    
    # VE: Task Energy Consumption (mJ/cycle) - energy per CPU cycle
    VE_map = {0: 6.0e-7, 1: 3.0e-7, 2: 2.0e-7, 3: 1.2e-7}
    
    # Create environment dictionary
    env_dict = create_environment_dict(
        locations_types=locations_types,
        DR_map=DR_map,
        DE_map=DE_map,
        VR_map=VR_map,
        VE_map=VE_map
    )
    
    # Create Environment object
    env = Environment.from_matrices(
        types=locations_types,
        DR_matrix=DR_map,
        DE_vector=DE_map,
        VR_vector=VR_map,
        VE_vector=VE_map
    )
    
    # ------------------------ OPTIMIZATION PARAMETERS -----------------------
    # Cost coefficients and mode as per the paper
    params = {
    "CT": 0.2,       # Cost per unit time (Eq. 1)
    "CE": 1.20,      # Cost per unit energy (Eq. 2)
    "delta_t": 1,    # Weight for time cost (1=enabled, 0=disabled)
    "delta_e": 1,    # Weight for energy cost (1=enabled, 0=disabled)
    "fixed_locations": {1: 0},  # Task 1 fixed on IoT
}
    # Note: delta_t=1, delta_e=1 → Balanced Mode
    #       delta_t=1, delta_e=0 → Low Latency Mode
    #       delta_t=0, delta_e=1 → Low Power Mode
    
    # ========================================================================
    # RUN AGENTIC WORKFLOW
    # ========================================================================
    
    result = run_workflow(
        "Find optimal offloading policy for this edge-cloud task offloading scenario", 
        {
            "env": env_dict,  # Pass environment as dict
            "workflow": wf.to_experiment_dict(),  # Pass workflow as dict
            "params": params
        },
        log_file="agent_trace_detailed.txt"
    )

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
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