import os
from langgraph.graph import StateGraph, END
from agents.planner import PlannerAgent
from agents.evaluator import EvaluatorAgent
from agents.output import OutputAgent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def build_agentic_workflow():
    workflow = StateGraph()

    planner = PlannerAgent(GEMINI_API_KEY)
    evaluator = EvaluatorAgent(GEMINI_API_KEY)
    output = OutputAgent(GEMINI_API_KEY)

    # Add graph nodes
    workflow.add_node("planner", planner.run)
    workflow.add_node("evaluator", evaluator.run)
    workflow.add_node("output", output.run)

    # Define edges
    workflow.add_edge("planner", "evaluator")
    workflow.add_edge("evaluator", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


def run_workflow(task_description: str, environment_state: dict):
    workflow = build_agentic_workflow()
    result = workflow.invoke({
        "planner": {"task_description": task_description},
        "evaluator": {"environment_state": environment_state}
    })
    return result


if __name__ == "__main__":
    env_state = {"bandwidth": 50, "cpu_load": 0.7, "latency": 30}
    result = run_workflow("Optimize multi-task offloading policy.", env_state)
    print(result)
