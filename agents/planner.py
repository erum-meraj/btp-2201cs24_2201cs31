# agents/planner_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Planner agent (ReAct style) — generates a task offloading plan."""

    def __init__(self, api_key: str):
        super().__init__(api_key)

        self.prompt_template = PromptTemplate.from_template("""
        You are the Planner Agent in a multi-agent system for task offloading.
        Your job is to reason step-by-step (ReAct style) and create a high-level plan
        for how to evaluate and optimize task placement across edge and cloud nodes.

        Consider:
        - Task dependencies (DAG)
        - Network latency
        - Energy and cost tradeoffs

        Respond with a structured plan in plain text.

        Example:
        Step 1: Analyze current environment metrics.
        Step 2: Call evaluator to compute utilities for each task placement.
        Step 3: Update policy if suboptimal.
        Step 4: Send refined plan to OutputAgent.

        Context:
        {context}
        """)

    def create_plan(self, context: dict):
        prompt = self.prompt_template.format(context=context)
        return self.think(prompt)
