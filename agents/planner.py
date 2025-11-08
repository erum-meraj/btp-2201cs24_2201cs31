# agents/planner.py - UPDATED with logging and complete environment details
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from agents.base_agent import BaseAgent
import json


class PlannerAgent(BaseAgent):
    """Planner agent with Chain-of-Thought reasoning for task offloading."""

    def __init__(self, api_key: str, log_file: str = "agent_trace.txt"):
        super().__init__(api_key)
        self.log_file = log_file

        self.prompt_template = PromptTemplate.from_template("""
You are the Planner Agent in a multi-agent system for task offloading optimization.

Your job is to analyze the task offloading problem and create a comprehensive plan using Chain-of-Thought reasoning.

## Complete Environment Details:
{env_details}

## Workflow Structure:
{workflow_details}

## Optimization Parameters:
{params}

## Your Task:
Analyze this edge-cloud offloading scenario step-by-step:

1. **Environment Analysis**: What are the key characteristics of the environment?
   - How many nodes/locations are available?
   - What are the network characteristics (bandwidth, latency)?
   - What are the energy/compute constraints?

2. **Workflow Analysis**: What does the task dependency structure tell us?
   - How many tasks need to be scheduled?
   - What are the critical paths in the DAG?
   - Which tasks have high computational or data transfer requirements?

3. **Constraint Identification**: What are the key constraints and trade-offs?
   - Energy vs. latency trade-offs
   - Local execution vs. remote offloading
   - Edge vs. cloud placement considerations

4. **Strategy Formulation**: What approach should the evaluator take?
   - Should we prioritize certain tasks for offloading?
   - Which placement patterns are likely optimal?
   - What heuristics can guide the search?

5. **Optimization Goals**: What metrics matter most?
   - Time minimization (low latency mode)
   - Energy minimization (low power mode)
   - Balanced optimization

Provide a structured, detailed plan that will guide the evaluator agent.
""")

    def _format_env_details(self, env: dict):
        """Format environment details for the prompt."""
        details = []
        
        # Network topology
        dr = env.get('DR', env.get('DR_pair', {}))
        if dr:
            details.append("Network Data Rates (DR):")
            for (src, dst), rate in sorted(dr.items()):
                details.append(f"  Link ({src} → {dst}): {rate:.2e} bits/sec")
        
        # Energy coefficients
        if 'DE' in env:
            details.append("\nEnergy Coefficients (DE):")
            for loc, coeff in sorted(env['DE'].items()):
                details.append(f"  Location {loc}: {coeff:.4f} J/cycle")
        
        # Computation rates
        if 'VR' in env:
            details.append("\nComputation Rates (VR):")
            for loc, rate in sorted(env['VR'].items()):
                details.append(f"  Location {loc}: {rate:.2e} cycles/sec")
        
        # Energy for transmission
        if 'VE' in env:
            details.append("\nTransmission Energy (VE):")
            for loc, energy in sorted(env['VE'].items()):
                details.append(f"  Location {loc}: {energy:.4e} J/bit")
        
        return "\n".join(details)
    
    def _format_workflow_details(self, workflow: dict):
        """Format workflow details for the prompt."""
        tasks = workflow.get('tasks', [])
        details = [f"Total Tasks: {len(tasks)}\n"]
        
        for task in tasks:
            task_id = task.get('task_id', '?')
            size = task.get('size', 0)
            deps = task.get('dependencies', {})
            
            details.append(f"Task {task_id}:")
            details.append(f"  Size: {size} MB")
            if deps:
                details.append(f"  Dependencies: {deps}")
            else:
                details.append("  Dependencies: None")
        
        return "\n".join(details)

    def create_plan(self, env: dict, workflow: dict, params: dict):
        """Create a detailed plan using Chain-of-Thought reasoning."""
        
        env_details = self._format_env_details(env)
        workflow_details = self._format_workflow_details(workflow)
        params_str = json.dumps(params, indent=2)
        
        prompt = self.prompt_template.format(
            env_details=env_details,
            workflow_details=workflow_details,
            params=params_str
        )
        
        # Log the prompt
        self._log_interaction("PLANNER", prompt, None, "PROMPT")
        
        # Use CoT reasoning
        result = self.think_with_cot(prompt, return_reasoning=True)
        if isinstance(result, dict):
            reasoning = result.get('reasoning', 'No reasoning provided')
            answer = result.get('answer', 'No answer provided')
        else:
            reasoning = str(result)
            answer = str(result)
        
        full_response = f"""
## Reasoning Process:
{reasoning}

## Execution Plan:
{answer}
"""
        
        # Log the response
        self._log_interaction("PLANNER", None, full_response, "RESPONSE")
        
        return full_response

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
        """
        The PlannerAgent generates a plan based on the environment,
        using Chain-of-Thought reasoning for better decision-making.
        """
        env = state.get("env", {})
        workflow = state.get("workflow", {})
        params = state.get("params", {})
        
        plan = self.create_plan(env, workflow, params)

        new_state = dict(state)
        new_state["plan"] = plan
        
        print("\n" + "="*60)
        print("PLANNER OUTPUT (with CoT reasoning):")
        print("="*60)
        print(plan[:500] + "..." if len(plan) > 500 else plan)
        print("="*60 + "\n")
        
        return new_state