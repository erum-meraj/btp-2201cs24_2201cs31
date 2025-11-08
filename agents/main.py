# main.py - UPDATED with comprehensive logging
import os, json, dotenv
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional, List
from core.workflow import Workflow, Task

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
        dr = env.get('DR', env.get('DR_pair', {}))
        if dr:
            f.write("\nNetwork Data Rates (DR):\n")
            f.write("-" * 40 + "\n")
            for (src, dst), rate in sorted(dr.items()):
                f.write(f"  Link ({src} → {dst}): {rate:.2e} bits/sec\n")
        
        # Energy coefficients
        if env.get('DE'):
            f.write("\nEnergy Coefficients per Location (DE):\n")
            f.write("-" * 40 + "\n")
            for loc, coeff in sorted(env['DE'].items()):
                f.write(f"  Location {loc}: {coeff:.4f} J/cycle\n")
        
        # Computation rates
        if env.get('VR'):
            f.write("\nComputation Rates per Location (VR):\n")
            f.write("-" * 40 + "\n")
            for loc, rate in sorted(env['VR'].items()):
                f.write(f"  Location {loc}: {rate:.2e} cycles/sec\n")
        
        # Transmission energy
        if env.get('VE'):
            f.write("\nEnergy per Bit Transmission (VE):\n")
            f.write("-" * 40 + "\n")
            for loc, energy in sorted(env['VE'].items()):
                f.write(f"  Location {loc}: {energy:.4e} J/bit\n")
        
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
        tasks = workflow.get('tasks', [])
        
        f.write(f"\nTotal Tasks: {len(tasks)}\n")
        f.write("-" * 40 + "\n")
        
        for task in tasks:
            task_id = task.get('task_id', 'Unknown')
            size = task.get('size', 0)
            deps = task.get('dependencies', {})
            
            f.write(f"\nTask {task_id}:\n")
            f.write(f"  Size: {size} MB\n")
            f.write(f"  Dependencies: {deps if deps else 'None'}\n")
            
            if deps:
                f.write("  Dependency Details:\n")
                for dep_id, data_size in deps.items():
                    f.write(f"    - Depends on Task {dep_id}: {data_size} MB data transfer\n")
        
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
      - env: environment parameters
      - workflow: workflow dict (from Workflow.to_dict())
      - params: optional evaluator parameters
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

if __name__ == "__main__":
    from core.network import Network, Node
    from core.environment import Environment
    from core.workflow import Workflow, Task
    import json

    network = Network()
    network.add_node(Node(0, 'edge', compute_power=10e9, energy_coeff=0.5))
    network.add_node(Node(1, 'cloud', compute_power=50e9, energy_coeff=0.2))
    network.add_link(0, 1, bandwidth=10e6, delay=0.01)
    network.add_link(1, 0, bandwidth=10e6, delay=0.01)
    network.add_link(0, 0, bandwidth=10e6, delay=0.0)
    network.add_link(1, 1, bandwidth=10e6, delay=0.0)

    env = Environment(network)
    env.randomize(seed=42)

    tasks = [
        Task(0, size=5.0, dependencies={}),
        Task(1, size=10.0, dependencies={0: 2.0}),
        Task(2, size=8.0, dependencies={1: 1.0})
    ]
    wf = Workflow(tasks)

    result = run_workflow(
        "Find optimal offloading policy for this edge-cloud task offloading scenario", 
        {
            "env": env.get_all_parameters(),
            "workflow": wf.to_dict(),
            "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1}
        },
        log_file="agent_trace_detailed.txt"
    )

    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps(result.get("output", {}), indent=2))