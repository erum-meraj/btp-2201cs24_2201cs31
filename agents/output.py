# agents/output_agent.py
import json
from agents.base_agent import BaseAgent

class OutputAgent(BaseAgent):
    def format_output(self, plan: str, evaluation: str, optimal_policy, workflow_data: dict = None):
        policy_str = str(optimal_policy) if optimal_policy else "[]"
        
        task_mapping = ""
        if optimal_policy and workflow_data:
            tasks = workflow_data.get('tasks', [])
            task_mapping = "\n".join([
                f"  Task {i}: {self._get_location_name(loc)} (location {loc})"
                for i, loc in enumerate(optimal_policy)
                if i < len(tasks)
            ])
        
        prompt = f"""
You are the Output Agent providing final recommendations for task offloading.

## Planner's Analysis:
{plan[:300]}...

## Evaluation Result:
{evaluation}

## Optimal Policy Found:
{policy_str}

## Task-to-Location Mapping:
{task_mapping if task_mapping else "No valid policy found"}

Using Chain-of-Thought reasoning, explain:

1. **Why is this policy optimal?**
   - What makes this placement better than alternatives?
   - How does it balance competing objectives?

2. **What are the key benefits?**
   - Performance improvements
   - Energy savings
   - Cost reductions

3. **What are potential risks or considerations?**
   - Network dependencies
   - Failure scenarios
   - Resource availability

4. **Implementation recommendations**
   - Monitoring requirements
   - Fallback strategies

Provide your explanation in a clear, structured format.
"""
        
        result = self.think_with_cot(prompt, return_reasoning=True)
        
        # Construct comprehensive output
        output = {
            "plan_summary": plan[:500] + ("..." if len(plan) > 500 else ""),
            "evaluation_summary": evaluation,
            "recommended_policy": list(optimal_policy),
            "task_mapping": task_mapping,
            "confidence": "High" if optimal_policy else "Low",
            "reasoning": result['reasoning'],
            "explanation": result['answer']
        }
        
        return json.dumps(output, indent=2)

    def _get_location_name(self, location: int) -> str:
        """Convert location ID to human-readable name."""
        if location == 0:
            return "Local Device"
        elif location == 1:
            return "Edge Server"
        else:
            return f"Cloud Server {location-1}"

    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        optimal_policy = state.get("optimal_policy", [])
        workflow_data = state.get("workflow", {})
        
        print(f"DEBUG (OutputAgent): Received optimal_policy = {optimal_policy}")
        
        output = self.format_output(plan, evaluation, optimal_policy, workflow_data)
        
        print("\n" + "="*60)
        print("FINAL OUTPUT (with CoT explanation):")
        print("="*60)
        print(json.dumps(json.loads(output), indent=2))
        print("="*60 + "\n")
        
        return {
            **state,
            "output": output
        }