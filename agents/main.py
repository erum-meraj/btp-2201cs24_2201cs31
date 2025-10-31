import os, json, dotenv
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional
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

# def run_workflow(task_description: str, environment_state: dict):
#     workflow = build_agentic_workflow()
#     result = workflow.invoke({
#         "query": task_description,
#         "env": environment_state
#     })

#     final_output = result.get("output", {})

#     # Save to Markdown
#     output_path = os.path.join(os.path.dirname(__file__), "result.md")
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("# 🧠 Agentic Workflow Result\n\n")
#         f.write(f"**Task:** {task_description}\n\n")
#         f.write("## 🌍 Environment\n")
#         f.write(f"```json\n{json.dumps(environment_state, indent=2)}\n```\n\n")

#         f.write("## 🪄 Final Output (Structured JSON)\n")
#         f.write(f"```json\n{json.dumps(final_output, indent=2)}\n```\n\n")

#         # If JSON contains specific keys, show readable summary
#         if isinstance(final_output, dict):
#             plan = final_output.get("plan_summary", "")
#             evaluation = final_output.get("evaluation_summary", "")
#             policy = final_output.get("recommended_policy", "")
#             confidence = final_output.get("confidence", "")

#             if plan:
#                 f.write("## 🧩 Plan Summary\n")
#                 f.write(plan + "\n\n")
#             if evaluation:
#                 f.write("## 📊 Evaluation Summary\n")
#                 f.write(evaluation + "\n\n")
#             if policy:
#                 f.write("## ✅ Recommended Policy\n")
#                 f.write(policy + "\n\n")
#             if confidence:
#                 f.write(f"**Confidence Level:** {confidence}\n")

#     print(f"✅ Result saved to {output_path}")
#     return result


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
        "workflow": wf.to_dict(),              # ✅ now included
        "params": {"CT": 0.2, "CE": 1.34, "delta_t": 1, "delta_e": 1}
    })

    print(json.dumps(result.get("output", {}), indent=2))
