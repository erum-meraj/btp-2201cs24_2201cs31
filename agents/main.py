import os, json, dotenv
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional, List
from core.workflow import Workflow, Task

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Agentic State Definition
# -----------------------------
class AgenticState(TypedDict, total=False):
    query: str
    env: dict
    workflow: dict          # ✅ Add this
    params: Optional[dict]  # ✅ Add this
    plan: Optional[str]
    evaluation: Optional[str]
    output: Optional[dict]
    optimal_policy: Optional[List[int]]



# -----------------------------
# Build the Agentic Workflow
# -----------------------------
def build_agentic_workflow():
    workflow = StateGraph(AgenticState)

    planner = PlannerAgent(GEMINI_API_KEY) # type: ignore
    evaluator = EvaluatorAgent(GEMINI_API_KEY) # type: ignore
    output = OutputAgent(GEMINI_API_KEY)  # type: ignore

    # Add nodes
    workflow.add_node("planner", planner.run) # type: ignore
    workflow.add_node("evaluator", evaluator.run) # type: ignore
    workflow.add_node("output", output.run) # type: ignore

    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "evaluator")
    workflow.add_edge("evaluator", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


# -----------------------------
# Run the Workflow
# -----------------------------
def run_workflow(task_description: str, state_data: dict):
    """
    state_data should include:
      - env: environment parameters
      - workflow: workflow dict (from Workflow.to_dict())
      - params: optional evaluator parameters
    """
    workflow = build_agentic_workflow()
    result = workflow.invoke({
        "query": task_description,
        **state_data  
    })

    final_output = result.get("output", {})
    ...
    return result

# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    from core.network import Network, Node
    from core.environment import Environment
    from core.workflow import Workflow, Task
    import json

    # Step 1: Build network
    network = Network()
    network.add_node(Node(0, 'edge', compute_power=10e9, energy_coeff=0.5))
    network.add_node(Node(1, 'cloud', compute_power=50e9, energy_coeff=0.2))
    network.add_link(0, 1, bandwidth=10e6, delay=0.01)
    network.add_link(1, 0, bandwidth=10e6, delay=0.01)
    # Add self-links to allow local processing (important!)
    network.add_link(0, 0, bandwidth=10e6, delay=0.0)
    network.add_link(1, 1, bandwidth=10e6, delay=0.0)


    # Step 2: Create environment
    env = Environment(network)
    env.randomize(seed=42)

    # Step 3: Define a small workflow (example with 3 tasks)
    tasks = [
        Task(0, size=5.0, dependencies={}),
        Task(1, size=10.0, dependencies={0: 2.0}),
        Task(2, size=8.0, dependencies={1: 1.0})
    ]
    wf = Workflow(tasks)

    # Step 4: Pass workflow + environment to run_workflow
    result = run_workflow("Find optimal offloading policy", {
        "env": env.get_all_parameters(),
        "workflow": wf.to_dict(),
        "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1}
    })

    print(json.dumps(result.get("output", {}), indent=2))
