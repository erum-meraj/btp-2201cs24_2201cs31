# agents/evaluator_agent.py
"""

EvaluatorAgent:
 - wraps your UtilityEvaluator to score candidate placements
 - returns sorted results with per-candidate cost and optional per-task marginal improvements

"""

from typing import List, Dict, Any
from core.cost_eval import UtilityEvaluator
from core.workflow import Workflow
from core.environment import Environment

class EvaluatorAgent:
    def __init__(self, util: UtilityEvaluator = None):
        self.util = util or UtilityEvaluator()

    def score(self, workflow: Workflow, env: Environment, placement: List[int]) -> float:
        """
        Return the total_offloading_cost computed by UtilityEvaluator.total_offloading_cost.
        Expects env to expose DR_pair (pairwise DR), DE, VR, VE as attributes (see your Environment).
        """
        params = {
            "DR": getattr(env, "DR_pair", {}),
            "DE": getattr(env, "DE", {}),
            "VR": getattr(env, "VR", {}),
            "VE": getattr(env, "VE", {}),
        }
        return self.util.total_offloading_cost(workflow, placement, params)

    def score_candidates(self, workflow: Workflow, env: Environment, candidates: List[List[int]]) -> List[Dict[str, Any]]:
        results = []
        base_score = None
        for cand in candidates:
            cost = self.score(workflow, env, cand)
            results.append({"placement": cand, "cost": cost})
            if base_score is None:
                base_score = cost
        # sort ascending (lower cost better)
        results.sort(key=lambda x: (float('inf') if x["cost"] is None else x["cost"]))
        return results
