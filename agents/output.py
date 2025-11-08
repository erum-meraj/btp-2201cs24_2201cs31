# agents/output.py - UPDATED with logging and complete environment details
import json
from agents.base_agent import BaseAgent


class OutputAgent(BaseAgent):
    def __init__(self, api_key: str, log_file: str = "agent_trace.txt"):
        super().__init__(api_key)
        self.log_file = log_file

    def format_output(self, plan: str, evaluation: str, optimal_policy, 
                     workflow_data: dict = None, env: dict = None, params: dict = None):
        policy_str = str(optimal_policy) if optimal_policy else "[]"
        
        task_mapping = ""
        if optimal_policy and workflow_data:
            tasks = workflow_data.get('tasks', [])
            task_mapping = "\n".join([
                f"  Task {i}: {self._get_location_name(loc)} (location {loc})"
                for i, loc in enumerate(optimal_policy)
                if i < len(tasks)
            ])
        
        # Format environment details
        env_summary = self._format_env_summary(env) if env else "No environment details"
        params_str = json.dumps(params, indent=2) if params else "No parameters"
        
        prompt = f"""
You are the Output Agent providing final recommendations for task offloading.

## Complete Environment Configuration:
{env_summary}

## Optimization Parameters:
{params_str}

## Planner's Analysis:
{plan[:500]}...

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
   - How does it leverage the network topology and compute resources?

2. **What are the key benefits?**
   - Performance improvements (latency reduction)
   - Energy savings
   - Cost reductions
   - Resource utilization

3. **What are potential risks or considerations?**
   - Network dependencies and bottlenecks
   - Failure scenarios
   - Resource availability
   - Task dependencies impact

4. **Implementation recommendations**
   - Monitoring requirements
   - Fallback strategies
   - Scaling considerations

Provide your explanation in a clear, structured format that justifies the placement decision.
"""
        
        # Log the prompt
        self._log_interaction("OUTPUT", prompt, None, "PROMPT")
        
        result = self.think_with_cot(prompt, return_reasoning=True)
        
        full_response = f"REASONING:\n{result['reasoning']}\n\nEXPLANATION:\n{result['answer']}"
        
        # Log the response
        self._log_interaction("OUTPUT", None, full_response, "RESPONSE")
        
        # Construct comprehensive output
        output = {
            "plan_summary": plan[:500] + ("..." if len(plan) > 500 else ""),
            "evaluation_summary": evaluation,
            "recommended_policy": list(optimal_policy) if optimal_policy else [],
            "task_mapping": task_mapping,
            "confidence": "High" if optimal_policy else "Low",
            "reasoning": result['reasoning'],
            "explanation": result['answer']
        }
        
        return json.dumps(output, indent=2)

    def _format_env_summary(self, env: dict):
        """Format environment summary for prompt."""
        lines = []
        
        # Network data rates
        dr = env.get('DR', env.get('DR_pair', {}))
        if dr:
            lines.append("Network Topology:")
            for (src, dst), rate in sorted(dr.items()):
                lines.append(f"  Link ({src} → {dst}): {rate:.2e} bits/sec")
        
        # Energy coefficients
        if env.get('DE'):
            lines.append("\nEnergy Coefficients:")
            for loc, coeff in sorted(env['DE'].items()):
                lines.append(f"  Location {loc}: {coeff:.4f} J/cycle")
        
        # Computation rates
        if env.get('VR'):
            lines.append("\nComputation Rates:")
            for loc, rate in sorted(env['VR'].items()):
                lines.append(f"  Location {loc}: {rate:.2e} cycles/sec")
        
        # Transmission energy
        if env.get('VE'):
            lines.append("\nTransmission Energy:")
            for loc, energy in sorted(env['VE'].items()):
                lines.append(f"  Location {loc}: {energy:.4e} J/bit")
        
        return "\n".join(lines)

    def _get_location_name(self, location: int) -> str:
        """Convert location ID to human-readable name."""
        if location == 0:
            return "Local Device"
        elif location == 1:
            return "Edge Server"
        else:
            return f"Cloud Server {location-1}"

    def _log_interaction(self, agent: str, prompt: str, response: str, msg_type: str):
        """Log agent interactions to file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if msg_type == "PROMPT":
                f.write("\n" + "="*80 + "\n")
                f.write(f"{agent} AGENT - PROMPT\n")
                f.write("="*80 + "\n")
                f.write(prompt)
                f.write("\n" + "="*80 + "\n\n")
            elif msg_type == "RESPONSE":
                f.write("\n" + "="*80 + "\n")
                f.write(f"{agent} AGENT - RESPONSE\n")
                f.write("="*80 + "\n")
                f.write(response)
                f.write("\n" + "="*80 + "\n\n")

    def run(self, state: dict):
        plan = state.get("plan", "")
        evaluation = state.get("evaluation", "")
        optimal_policy = state.get("optimal_policy", [])
        workflow_data = state.get("workflow", {})
        env = state.get("env", {})
        params = state.get("params", {})
        
        output = self.format_output(plan, evaluation, optimal_policy, 
                                    workflow_data, env, params)
        
        print("\n" + "="*60)
        print("FINAL OUTPUT (with CoT explanation):")
        print("="*60)
        print(json.dumps(json.loads(output), indent=2))
        print("="*60 + "\n")
        
        return {
            **state,
            "output": output
        }