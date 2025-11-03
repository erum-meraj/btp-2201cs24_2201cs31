# agents/planner_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from agents.base_agent import BaseAgent
import json


class PlannerAgent(BaseAgent):
    """Planner agent with Chain-of-Thought reasoning for task offloading."""

    def __init__(self, api_key: str):
        super().__init__(api_key)

        self.prompt_template = PromptTemplate.from_template("""
You are the Planner Agent in a multi-agent system for task offloading optimization.

Your job is to analyze the task offloading problem and create a comprehensive plan using Chain-of-Thought reasoning.

## Problem Context:
{context}

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

    def create_plan(self, context: dict):
        """Create a detailed plan using Chain-of-Thought reasoning."""

        env_summary = self._summarize_environment(context)
        prompt = self.prompt_template.format(context=json.dumps(env_summary, indent=2))
        
        # Use CoT reasoning
        result = self.think_with_cot(prompt, return_reasoning=True)
        if isinstance(result, dict):
            reasoning = result.get('reasoning', 'No reasoning provided')
            answer = result.get('answer', 'No answer provided')
        else:
            reasoning = str(result)
            answer = str(result)
            
        plan = f"""
## Reasoning Process:
{reasoning}

## Execution Plan:
{answer}
"""
        return plan

    def _summarize_environment(self, context: dict):
        """Summarize environment for more focused reasoning."""
        summary = {}
        
        if 'DR' in context:
            dr = context['DR']
            locations = set()
            for (src, dst) in dr.keys():
                locations.add(src)
                locations.add(dst)
            summary['num_locations'] = len(locations)
            summary['locations'] = sorted(list(locations))
            summary['has_network_info'] = True
        
        if 'DE' in context:
            summary['energy_aware'] = True
            summary['num_energy_nodes'] = len(context['DE'])
        
        if 'VR' in context:
            summary['compute_aware'] = True
            summary['num_compute_nodes'] = len(context['VR'])
        
        return summary

    def run(self, state: dict):
        """
        The PlannerAgent generates a plan based on the environment,
        using Chain-of-Thought reasoning for better decision-making.
        """
        context = state.get("env") or {}
        workflow = state.get("workflow") or {}
        
        # Add workflow info to context
        if workflow:
            context['workflow_info'] = {
                'num_tasks': len(workflow.get('tasks', [])),
                'has_dependencies': any(
                    task.get('dependencies', {}) 
                    for task in workflow.get('tasks', [])
                )
            }
        
        plan = self.create_plan(context)

        new_state = dict(state)
        new_state["plan"] = plan

        # print("DEBUG (Planner): Forwarding keys =>", list(new_state.keys()))
        print("\n" + "="*60)
        print("PLANNER OUTPUT (with CoT reasoning):")
        print("="*60)
        print(plan[:500] + "..." if len(plan) > 500 else plan)
        print("="*60 + "\n")
        
        return new_state