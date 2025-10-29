# agents/output_agent.py
from agents.base_agent import BaseAgent

class OutputAgent(BaseAgent):
    """Formats final structured output."""

    def format_output(self, plan: str, evaluation: str):
        prompt = f"""
        You are the Output Agent. Structure the final output into a JSON-like schema.

        Plan Summary:
        {plan}

        Evaluation Summary:
        {evaluation}

        Output Format:
        {{
          "plan_summary": "...",
          "evaluation_summary": "...",
          "recommended_policy": "...",
          "confidence": "..."
        }}
        """

        return self.think(prompt)
    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        output = self.format_output(plan, evaluation)
        return {"plan": plan, "evaluation": evaluation, "output": output}