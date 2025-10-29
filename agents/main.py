import os, json, dotenv
from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent
from typing import TypedDict, Optional

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Agentic State Definition
# -----------------------------
class AgenticState(TypedDict, total=False):
    query: str
    env: dict
    plan: Optional[str]
    evaluation: Optional[str]
    output: Optional[dict]  # expecting a structured JSON output


# -----------------------------
# Build the Agentic Workflow
# -----------------------------
def build_agentic_workflow():
    workflow = StateGraph(AgenticState)

    planner = PlannerAgent(GEMINI_API_KEY)
    evaluator = EvaluatorAgent(GEMINI_API_KEY)
    output = OutputAgent(GEMINI_API_KEY)

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


# -----------------------------
# Run the Workflow
# -----------------------------
def run_workflow(task_description: str, environment_state: dict):
    workflow = build_agentic_workflow()
    result = workflow.invoke({
        "query": task_description,
        "env": environment_state
    })

    final_output = result.get("output", {})

    # Save to Markdown
    output_path = os.path.join(os.path.dirname(__file__), "result.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 🧠 Agentic Workflow Result\n\n")
        f.write(f"**Task:** {task_description}\n\n")
        f.write("## 🌍 Environment\n")
        f.write(f"```json\n{json.dumps(environment_state, indent=2)}\n```\n\n")

        f.write("## 🪄 Final Output (Structured JSON)\n")
        f.write(f"```json\n{json.dumps(final_output, indent=2)}\n```\n\n")

        # If JSON contains specific keys, show readable summary
        if isinstance(final_output, dict):
            plan = final_output.get("plan_summary", "")
            evaluation = final_output.get("evaluation_summary", "")
            policy = final_output.get("recommended_policy", "")
            confidence = final_output.get("confidence", "")

            if plan:
                f.write("## 🧩 Plan Summary\n")
                f.write(plan + "\n\n")
            if evaluation:
                f.write("## 📊 Evaluation Summary\n")
                f.write(evaluation + "\n\n")
            if policy:
                f.write("## ✅ Recommended Policy\n")
                f.write(policy + "\n\n")
            if confidence:
                f.write(f"**Confidence Level:** {confidence}\n")

    print(f"✅ Result saved to {output_path}")
    return result


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    env_state = {"bandwidth": 50, "cpu_load": 0.7, "latency": 30}
    result = run_workflow("Optimize multi-task offloading policy.", env_state)
    print(json.dumps(result.get("output", {}), indent=2))
