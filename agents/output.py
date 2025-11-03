# agents/output_agent.py
import json
import re
from agents.base_agent import BaseAgent

class OutputAgent(BaseAgent):
    """Formats final output with optimal policy."""

    def format_output(self, plan: str, evaluation: str, optimal_policy):
        # Convert policy to a clear string representation
        policy_str = str(optimal_policy) if optimal_policy else "[]"
        
        prompt = f"""
        You are the Output Agent. Provide the final output strictly following the research paper format.

        Plan Summary:
        {plan}

        Evaluation Summary:
        {evaluation}

        The optimal offloading policy vector (p) is: {policy_str}
        This means: Task 0 → Location {optimal_policy[0] if len(optimal_policy) > 0 else 'N/A'}
                    Task 1 → Location {optimal_policy[1] if len(optimal_policy) > 1 else 'N/A'}
                    Task 2 → Location {optimal_policy[2] if len(optimal_policy) > 2 else 'N/A'}

        Return ONLY a valid JSON object (no markdown formatting) with:
        {{
          "plan_summary": "Brief 2-3 sentence summary of the planning approach",
          "evaluation_summary": "The evaluation result exactly as provided",
          "recommended_policy": {policy_str},
          "confidence": "High" or "Low" based on whether a policy was found
        }}
        """
        
        response = self.think(prompt)
        
        # Try to extract JSON from the response (in case LLM wraps it in markdown)
        try:
            # Remove markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # Parse the JSON
            output = json.loads(json_str)
            
            # Ensure the policy is correct (override if LLM got it wrong)
            output["recommended_policy"] = list(optimal_policy)
            output["confidence"] = "High" if optimal_policy else "Low"
            
            return json.dumps(output, indent=2)
            
        except json.JSONDecodeError:
            # Fallback: construct the output directly if LLM fails to generate valid JSON
            output = {
                "plan_summary": "Task offloading plan generated successfully",
                "evaluation_summary": evaluation,
                "recommended_policy": list(optimal_policy),
                "confidence": "High" if optimal_policy else "Low"
            }
            return json.dumps(output, indent=2)

    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        optimal_policy = state.get("optimal_policy", [])
        
        # Debug: Print what we received
        print(f"DEBUG (OutputAgent): Received optimal_policy = {optimal_policy}")
        
        output = self.format_output(plan, evaluation, optimal_policy)
        return {"plan": plan, "evaluation": evaluation, "output": output}