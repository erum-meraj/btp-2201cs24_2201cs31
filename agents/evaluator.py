# agents/evaluator_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from core.cost_eval import UtilityEvaluator


class EvaluatorAgent:
    """Evaluator agent — uses Gemini for reasoning and a Utility tool for calculations."""

    def __init__(self, api_key: str):
        self.utility_tool = StructuredTool.from_function(
            func=UtilityEvaluator().total_offloading_cost,
            name="utility_evaluator",
            description="Computes utility metrics (latency, cost, and energy consumption) for a given offloading plan."
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.4,
        ).bind_tools([self.utility_tool])

    def evaluate_plan(self, plan: str, environment: dict):
        prompt = f"""
        You are the Evaluator Agent. You receive a plan from the Planner Agent.

        Plan:
        {plan}

        Evaluate this plan by reasoning through it and using the `utility_evaluator` tool
        when appropriate to compute metrics for task offloading.

        Environment Context:
        {environment}

        Return a structured summary of the evaluation, including utility values and suggestions for improvement.
        """

        response = self.llm.invoke(prompt)
        return response.content
