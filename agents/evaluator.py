import itertools
import math
from core.workflow import Workflow
from core.cost_eval import UtilityEvaluator
from agents.base_agent import BaseAgent
import json


class EvaluatorAgent(BaseAgent):
    """Evaluator agent with CoT-guided heuristic search."""

    def __init__(self, api_key: str ):
        super().__init__(api_key)
        self.evaluator = UtilityEvaluator()

    def _normalize_env(self, environment: dict) -> dict:
        """Convert environment dict to expected format."""
        env = {}

        for k in ("DE", "VE", "VR"):
            if k in environment:
                env[k] = environment[k]
            elif k.lower() in environment:
                env[k] = environment[k.lower()]
            else:
                env[k] = {}

        if "DR" in environment:
            env["DR"] = environment["DR"]
        elif "DR_pair" in environment:
            env["DR"] = environment["DR_pair"]
        elif "dr_pair" in environment:
            env["DR"] = environment["dr_pair"]
        else:
            env["DR"] = environment.get("DR_pair", {})

        return env

    def get_llm_guided_heuristics(self, workflow_data: dict, environment: dict, plan: str):
        """Use LLM with CoT to generate heuristic guidance for policy search."""
        
        workflow = Workflow.from_dict(workflow_data)
        n_tasks = len(workflow.tasks)

        env = self._normalize_env(environment)
        locs = set()
        for d in (env.get("DE", {}), env.get("VE", {}), env.get("VR", {})):
            locs.update(d.keys())
        
        prompt = f"""
You are helping optimize task offloading decisions for an edge-cloud system.

## Workflow Information:
- Number of tasks: {n_tasks}
- Task dependencies: {json.dumps({t.task_id: list(t.dependencies.keys()) for t in workflow.tasks}, indent=2)}
- Task sizes: {json.dumps({t.task_id: t.size for t in workflow.tasks}, indent=2)}

## Available Locations:
{sorted(list(locs))}
- Location 0: Local (IoT device)
- Location 1+: Remote (Edge/Cloud servers)

## Planner's Analysis:
{plan[:500]}

Based on the above information, suggest intelligent heuristics for task placement:

1. Which tasks should definitely be offloaded (high compute, low data transfer)?
2. Which tasks should stay local (low compute, high data dependency)?
3. What placement patterns make sense given the DAG structure?
4. Should we group dependent tasks on the same node?

Provide 3-5 specific candidate placement policies to evaluate.
Format: For each policy, list the location for each task [task0_loc, task1_loc, task2_loc, ...]
"""
        
        result = self.think_with_cot(prompt, return_reasoning=True)
        
        print("\n" + "="*60)
        print("LLM HEURISTIC REASONING:")
        print("="*60)
        print(result['reasoning'][:400] + "...") # type: ignore
        print("="*60 + "\n")

        policies = self._parse_policies_from_text(result['answer'], n_tasks, len(locs)) # type: ignore
        
        return policies

    def _parse_policies_from_text(self, text: str, n_tasks: int, n_locations: int):
        """Extract policy suggestions from LLM text output."""
        import re
        
        policies = []

        pattern = r'[\[\(](\d+(?:\s*,\s*\d+)*)[\]\)]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                policy = [int(x.strip()) for x in match.split(',')]
                if len(policy) == n_tasks and all(0 <= loc < n_locations for loc in policy):
                    policies.append(tuple(policy))
            except:
                continue

        seen = set()
        unique_policies = []
        for p in policies:
            if p not in seen:
                seen.add(p)
                unique_policies.append(p)
        
        return unique_policies[:5]  # Return top 5 suggestions

    def find_best_policy(self, workflow_data: dict, environment: dict, params: dict, plan: str = ""):
        """
        Search for optimal placement using CoT-guided heuristics + exhaustive search.
        """
        if workflow_data is None:
            raise ValueError("workflow_data is None")

        workflow = Workflow.from_dict(workflow_data)
        n_tasks = len(workflow.tasks)

        env = self._normalize_env(environment or {})

        locs = set()
        for d in (env.get("DE", {}), env.get("VE", {}), env.get("VR", {})):
            locs.update(d.keys())
        for k in env.get("DR", {}).keys():
            try:
                src, dst = k
                locs.add(src)
                locs.add(dst)
            except Exception:
                pass

        n_locations = max(locs) + 1 if locs else 2

        # Prepare evaluator params
        evaluator_params = {
            "DE": env.get("DE", {}),
            "VE": env.get("VE", {}),
            "VR": env.get("VR", {}),
            "DR": env.get("DR", {})
        }
        if params:
            evaluator_params.update(params)

        print("\nUsing Chain-of-Thought to generate intelligent candidate policies...")
        llm_candidates = self.get_llm_guided_heuristics(workflow_data, environment, plan)
        
        # Generate additional systematic candidates
        systematic_candidates = []
        systematic_candidates.append(tuple(0 for _ in range(n_tasks)))
        if n_locations > 1:
            systematic_candidates.append(tuple(1 for _ in range(n_tasks)))
            for start in range(min(n_locations, 3)):
                cand = tuple((start + i) % n_locations for i in range(n_tasks))
                systematic_candidates.append(cand)
        
        # Combine candidates (LLM suggestions first, then systematic)
        candidates = llm_candidates + [c for c in systematic_candidates if c not in llm_candidates]
        
        # If problem is small enough, also do exhaustive search
        max_exhaustive = 10000
        total_combos = n_locations ** n_tasks
        if total_combos <= max_exhaustive:
            print(f"Problem size allows exhaustive search ({total_combos} combinations)")
            all_candidates = list(itertools.product(range(n_locations), repeat=n_tasks))
            candidates = list(set(candidates + all_candidates))
        else:
            print(f" Problem too large for exhaustive search ({total_combos} combinations)")
            print(f"  Using {len(candidates)} LLM-guided + heuristic candidates")

        print(f"\nEvaluating {len(candidates)} candidate policies...")
        
        best_policy = None
        best_cost = float("inf")
        evaluated = 0
        skipped = 0

        for placement in candidates:
            try:
                cost = self.evaluator.total_offloading_cost(workflow, list(placement), evaluator_params)
                evaluated += 1

                if cost is None or (isinstance(cost, float) and math.isinf(cost)):
                    skipped += 1
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_policy = tuple(placement)
                    print(f"  ✓ New best: {best_policy} with cost {best_cost:.6f}")

            except KeyError as ke:
                skipped += 1
            except Exception as e:
                skipped += 1

        return {
            "best_policy": best_policy,
            "best_cost": best_cost,
            "evaluated": evaluated,
            "skipped": skipped
        }

    def run(self, state: dict):
        workflow_data = state.get("workflow")
        environment = state.get("env", {})
        params = state.get("params", {})
        plan = state.get("plan", "")
        
        print("DEBUG (Evaluator): Keys in state =>", list(state.keys()))

        if not isinstance(workflow_data, dict):
            raise ValueError("workflow_data must be a dictionary")

        result = self.find_best_policy(workflow_data, environment, params, plan)

        if result["best_policy"] is None:
            evaluation = f"No finite-cost policy found. evaluated={result['evaluated']}, skipped={result['skipped']}"
            optimal_policy = []
        else:
            evaluation = f"Best policy found with total cost = {result['best_cost']}"
            optimal_policy = list(result['best_policy'])

        print(f"\nDEBUG (Evaluator): Returning optimal_policy = {optimal_policy}")

        return {
            **state,
            "evaluation": evaluation,
            "optimal_policy": optimal_policy
        }