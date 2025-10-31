import itertools
from core.workflow import Workflow
from core.cost_eval import UtilityEvaluator

class EvaluatorAgent:
    """Evaluator agent — computes utility and finds optimal offloading policy."""

    def __init__(self, api_key: str):
        self.evaluator = UtilityEvaluator()
        self.api_key = api_key

    def find_best_policy(self, workflow_data: dict, environment: dict, params: dict):
        """
        Search over all possible placements (brute-force or sampled)
        to find the placement vector p that minimizes total offloading cost U(w, p).
        """
        workflow = Workflow.from_dict(workflow_data)
        n_tasks = len(workflow.tasks)
        n_locations = len(environment.get("DE", {}))  # number of available locations

        # Generate all possible placement combinations (limited for small tasks)
        all_policies = list(itertools.product(range(n_locations), repeat=n_tasks))

        best_policy, best_cost = None, float("inf")
        for placement in all_policies:
            try:
                cost = self.evaluator.total_offloading_cost(workflow, placement, environment)
                if cost < best_cost:
                    best_cost = cost
                    best_policy = placement
            except Exception as e:
                print(f"⚠️ Skipped invalid policy {placement}: {e}")

        return {"best_policy": best_policy, "best_cost": best_cost}

    def run(self, state: dict):
        workflow_data = state.get("workflow")
        environment = state.get("env", {})
        params = state.get("params", {})
        plan = state.get("plan", "")
        print("DEBUG: Keys in state =>", list(state.keys()))

        result = self.find_best_policy(workflow_data, environment, params)
        return {
            "plan": plan,
            "evaluation": f"Best policy found with total cost = {result['best_cost']}",
            "optimal_policy": result["best_policy"]
        }
