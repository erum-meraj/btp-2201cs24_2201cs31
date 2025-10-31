import itertools
import math
from core.workflow import Workflow
from core.cost_eval import UtilityEvaluator

class EvaluatorAgent:
    """Evaluator agent — computes utility and finds optimal offloading policy."""

    def __init__(self, api_key: str = None):
        self.evaluator = UtilityEvaluator()
        self.api_key = api_key

    def _normalize_env(self, environment: dict) -> dict:
        """
        Convert a user-provided environment dict into the keys expected by UtilityEvaluator.
        Accept common variants: 'DR_pair' or 'DR', and ensure DE, VE, VR are present.
        """
        env = {}

        # energy per byte (DE), energy per compute (VE), time per compute (VR)
        for k in ("DE", "VE", "VR"):
            if k in environment:
                env[k] = environment[k]
            elif k.lower() in environment:
                env[k] = environment[k.lower()]
            else:
                env[k] = {}

        # DR may be provided as 'DR', 'DR_pair' or inside nested 'DR'
        if "DR" in environment:
            env["DR"] = environment["DR"]
        elif "DR_pair" in environment:
            env["DR"] = environment["DR_pair"]
        elif "dr_pair" in environment:
            env["DR"] = environment["dr_pair"]
        else:
            # fallback: try to compute from network if provided, else empty dict
            env["DR"] = environment.get("DR_pair", {})

        return env

    def find_best_policy(self, workflow_data: dict, environment: dict, params: dict):
        """
        Search over all possible placements (brute-force or sampled)
        to find the placement vector p that minimizes total offloading cost U(w, p).

        Returns a dict:
          { "best_policy": tuple or None,
            "best_cost": float,
            "skipped": int,
            "evaluated": int }
        """
        if workflow_data is None:
            raise ValueError("workflow_data is None")

        workflow = Workflow.from_dict(workflow_data)
        n_tasks = len(workflow.tasks)

        env = self._normalize_env(environment or {})
        # number of locations is determined by DE keys, or VE or VR (take union)
        locs = set()
        for d in (env.get("DE", {}), env.get("VE", {}), env.get("VR", {})):
            locs.update(d.keys())
        # fallback: if user supplied DR mapping, include its node ids
        for k in env.get("DR", {}).keys():
            try:
                src, dst = k
                locs.add(src); locs.add(dst)
            except Exception:
                pass

        if len(locs) == 0:
            # if no explicit nodes found, assume 2 locations (0 local and 1 remote) as default
            n_locations = 2
        else:
            n_locations = max(locs) + 1  # assume 0..max_node_id

        # prepare params object expected by UtilityEvaluator.total_offloading_cost
        # Many implementations expect a dict like {"DE":..., "VE":..., "VR":..., "DR":...}
        evaluator_params = {
            "DE": env.get("DE", {}),
            "VE": env.get("VE", {}),
            "VR": env.get("VR", {}),
            "DR_pair": env.get("DR", {})  # keep both names if evaluator expects either
        }
        # allow caller-provided overrides (params)
        if params:
            evaluator_params.update(params)

        # iterate placements (brute-force); guard for explosion by limiting total combos
        max_combinations = 5_000_000  # safety cap
        total_combos = (n_locations ** n_tasks)
        if total_combos > max_combinations:
            # if too many, sample heuristically: try p0 (all local), p1 (all remote 1), and a few random
            # but still try some deterministic candidates
            candidates = []
            candidates.append(tuple(0 for _ in range(n_tasks)))  # all local
            if n_locations > 1:
                candidates.append(tuple(1 for _ in range(n_tasks)))  # all remote 1
            # try round-robin placements for some diversity
            for start in range(min(n_locations, 4)):
                cand = tuple((start + i) % n_locations for i in range(n_tasks))
                candidates.append(cand)
        else:
            candidates = list(itertools.product(range(n_locations), repeat=n_tasks))

        best_policy = None
        best_cost = float("inf")
        evaluated = 0
        skipped = 0

        for placement in candidates:
            try:
                # UtilityEvaluator API: try common signatures.
                # Some implementations expect (workflow, placement, DE, VE, DR_pair, VR) or (workflow, placement, params)
                # We'll try the total_offloading_cost(workflow, placement, evaluator_params) call first.
                cost = None
                try:
                    cost = self.evaluator.total_offloading_cost(workflow, placement, evaluator_params)
                except TypeError:
                    # fallback attempt: pass individual components if signature differs
                    cost = self.evaluator.total_offloading_cost(
                        workflow,
                        list(placement),
                        evaluator_params.get("DE", {}),
                        evaluator_params.get("VE", {}),
                        evaluator_params.get("DR_pair", {}),
                        evaluator_params.get("VR", {})
                    )

                evaluated += 1

                # skip infinite costs (unreachable links)
                if cost is None or (isinstance(cost, float) and math.isinf(cost)):
                    skipped += 1
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_policy = tuple(placement)

            except KeyError as ke:
                skipped += 1
                # KeyError messages sometimes contain the missing key name (e.g., 'DR')
                print(f"⚠️ Skipped invalid policy {placement}: {ke}")
            except Exception as e:
                skipped += 1
                print(f"⚠️ Skipped invalid policy {placement}: {e}")

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
        print("DEBUG: Keys in state =>", list(state.keys()))

        result = self.find_best_policy(workflow_data, environment, params)

        if result["best_policy"] is None:
            evaluation = f"No finite-cost policy found. evaluated={result['evaluated']}, skipped={result['skipped']}"
            optimal_policy = []
        else:
            evaluation = f"Best policy found with total cost = {result['best_cost']}"
            optimal_policy = list(result['best_policy'])

        return {
            "plan": plan,
            "evaluation": evaluation,
            "optimal_policy": optimal_policy
        }
