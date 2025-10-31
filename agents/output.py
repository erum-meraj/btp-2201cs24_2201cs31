# agents/output_agent.py
from agents.base_agent import BaseAgent

class OutputAgent(BaseAgent):
    """Formats final output with optimal policy."""

    def format_output(self, plan: str, evaluation: str, optimal_policy):
        prompt = f"""
        You are the Output Agent. Provide the final output strictly following the research paper format.

        Plan Summary:
        {plan}

        Evaluation Summary:
        {evaluation}

        The optimal offloading policy vector (p) should be expressed as:
        p = {{{', '.join(map(str, optimal_policy))}}}

        Return JSON with:
        {{
          "plan_summary": "...",
          "evaluation_summary": "...",
          "recommended_policy": "{list(optimal_policy)}",
          "confidence": "High"
        }}
        """
        return self.think(prompt)

    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        optimal_policy = state.get("optimal_policy", [])
        output = self.format_output(plan, evaluation, optimal_policy)
        return {"plan": plan, "evaluation": evaluation, "output": output}
