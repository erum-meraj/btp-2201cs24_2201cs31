# agents/evaluator.py - FULLY INTEGRATED with UtilityEvaluator and paper specifications
import itertools
import math
from core.workflow import Workflow
from core.environment import Environment
from core.cost_eval import UtilityEvaluator
from agents.base_agent import BaseAgent
import json


class EvaluatorAgent(BaseAgent):
    """Evaluator agent with CoT-guided heuristic search using UtilityEvaluator."""

    def __init__(self, api_key: str, log_file: str = "agent_trace.txt"):
        super().__init__(api_key)
        self.log_file = log_file

    def _create_environment(self, env_dict: dict) -> Environment:
        """Create Environment object from dictionary."""
        locations_types = env_dict.get('locations', {})
        DR_map = env_dict.get('DR', {})
        DE_map = env_dict.get('DE', {})
        VR_map = env_dict.get('VR', {})
        VE_map = env_dict.get('VE', {})
        
        return Environment.from_matrices(
            types=locations_types,
            DR_matrix=DR_map,
            DE_vector=DE_map,
            VR_vector=VR_map,
            VE_vector=VE_map
        )

    def _format_env_for_prompt(self, env_dict: dict):
        """Format environment details comprehensively for LLM."""
        details = []
        
        # Locations
        locations = env_dict.get('locations', {})
        if locations:
            details.append("Available Locations:")
            for loc_id, loc_type in sorted(locations.items()):
                details.append(f"  Location {loc_id}: {loc_type.upper()} (l_{loc_id})")
                if loc_id == 0:
                    details.append(f"    → IoT device (local execution, no offloading)")
                else:
                    details.append(f"    → Remote server (offloading target)")
            details.append("")
        
        # DR - Data Transfer Time
        dr = env_dict.get('DR', {})
        if dr:
            details.append("Data Transfer Characteristics DR(li, lj) [ms/byte]:")
            for (src, dst), rate in sorted(dr.items()):
                if src != dst:
                    latency_per_mb = rate * 1e6  # Convert to ms/MB
                    details.append(f"  {src}→{dst}: {rate:.6e} ms/byte ({latency_per_mb:.3f} ms/MB)")
            details.append("")
        
        # DE - Data Energy
        de = env_dict.get('DE', {})
        if de:
            details.append("Data Energy Consumption DE(li) [mJ/byte]:")
            for loc, coeff in sorted(de.items()):
                details.append(f"  Location {loc}: {coeff:.6e} mJ/byte")
            details.append("")
        
        # VR - Computation Speed
        vr = env_dict.get('VR', {})
        if vr:
            details.append("Task Execution Speed VR(li) [ms/cycle]:")
            for loc, rate in sorted(vr.items()):
                # Convert to effective GHz for readability
                ghz = 1.0 / (rate * 1e6) if rate > 0 else float('inf')
                details.append(f"  Location {loc}: {rate:.6e} ms/cycle (≈{ghz:.1f} GHz)")
            details.append("")
        
        # VE - Task Energy
        ve = env_dict.get('VE', {})
        if ve:
            details.append("Task Energy Consumption VE(li) [mJ/cycle]:")
            for loc, energy in sorted(ve.items()):
                details.append(f"  Location {loc}: {energy:.6e} mJ/cycle")
        
        return "\n".join(details)

    def get_llm_guided_heuristics(self, workflow_dict: dict, env_dict: dict, plan: str, params: dict):
        """Use LLM with CoT to generate heuristic guidance for policy search."""
        
        # Extract workflow details
        tasks = workflow_dict.get('tasks', {})
        edges = workflow_dict.get('edges', {})
        N = workflow_dict.get('N', 0)
        
        # Get available locations
        locations = env_dict.get('locations', {})
        n_locations = len(locations)
        
        env_details = self._format_env_for_prompt(env_dict)
        
        # Build task details with paper notation
        task_details = []
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            v_i = task_data.get('v', 0)
            
            # Find dependencies
            parents = [j for (j, k), _ in edges.items() if k == task_id]
            children_deps = [(k, edges[(task_id, k)]) for (j, k), _ in edges.items() if j == task_id]
            
            task_details.append(f"\nTask {task_id}:")
            task_details.append(f"  v_{task_id} = {v_i:.2e} CPU cycles")
            
            if parents:
                task_details.append(f"  Depends on: Tasks {parents}")
            
            if children_deps:
                task_details.append(f"  Data output to:")
                for k, d_ik in children_deps:
                    task_details.append(f"    Task {k}: d_{{{task_id},{k}}} = {d_ik:.2e} bytes")
        
        prompt = f"""
You are helping optimize task offloading decisions for an edge-cloud system following the paper's framework.

## Environment Configuration:
{env_details}

## Workflow DAG (N = {N} tasks):
{chr(10).join(task_details)}

## Optimization Parameters:
{json.dumps(params, indent=2)}

## Planner's Strategic Analysis:
{plan[:800]}

## Your Task:
Generate 3-5 intelligent candidate placement policies p = {{l_1, l_2, ..., l_{N}}} where:
- l_i = 0 means execute Task i locally on IoT device
- l_i ≥ 1 means offload Task i to remote server l_i

Consider these heuristics from the paper:

1. **Computation-Heavy Tasks**: High v_i with low data dependencies → Good candidates for offloading
   
2. **Data-Heavy Tasks**: Low v_i but high d_{{i,j}} → May be better local to avoid transfer overhead

3. **Dependent Task Co-location**: If d_{{i,j}} is large, placing Tasks i and j on same location reduces transfer

4. **Critical Path Optimization**: Tasks on critical path should be placed on fastest available locations

5. **Mode-Specific Strategy**:
   - Low Latency: Prioritize fastest execution (VR), tolerate data transfer
   - Low Power: Minimize energy (VE, DE), accept slower execution  
   - Balanced: Find sweet spot between speed and energy

Provide candidate policies as lists: [l_1, l_2, ..., l_{N}]

Example format:
- Policy 1: [0, 1, 1] - Task 1 local, Tasks 2-3 on Edge
- Policy 2: [1, 2, 2] - Task 1 on Edge, Tasks 2-3 on Cloud
"""
        
        # Log the prompt
        self._log_interaction("EVALUATOR", prompt, None, "PROMPT")
        
        result = self.think_with_cot(prompt, return_reasoning=True)
        
        full_response = f"REASONING:\n{result['reasoning']}\n\nCANDIDATE POLICIES:\n{result['answer']}"
        
        # Log the response
        self._log_interaction("EVALUATOR", None, full_response, "RESPONSE")
        
        print("\n" + "="*60)
        print("LLM HEURISTIC REASONING:")
        print("="*60)
        print(result['reasoning'][:400] + "...")
        print("="*60 + "\n")

        policies = self._parse_policies_from_text(result['answer'], N, n_locations)
        
        return policies

    def _parse_policies_from_text(self, text: str, n_tasks: int, n_locations: int):
        """Extract policy suggestions from LLM text output."""
        import re
        
        policies = []

        # Match patterns like [0, 1, 2] or (0, 1, 2)
        pattern = r'[\[\(](\d+(?:\s*,\s*\d+)*)[\]\)]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                policy = [int(x.strip()) for x in match.split(',')]
                if len(policy) == n_tasks and all(0 <= loc < n_locations for loc in policy):
                    policies.append(tuple(policy))
            except:
                continue

        # Remove duplicates while preserving order
        seen = set()
        unique_policies = []
        for p in policies:
            if p not in seen:
                seen.add(p)
                unique_policies.append(p)
        
        return unique_policies[:5]

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

    def find_best_policy(self, workflow_dict: dict, env_dict: dict, params: dict, plan: str = ""):
        """
        Search for optimal placement using CoT-guided heuristics + exhaustive search.
        Uses UtilityEvaluator to compute offloading costs following the paper's equations.
        """
        if workflow_dict is None:
            raise ValueError("workflow_dict is None")

        # Create workflow and environment objects
        workflow = Workflow.from_experiment_dict(workflow_dict)
        env = self._create_environment(env_dict)
        
        N = workflow.N
        n_locations = len(env_dict.get('locations', {}))

        # Create evaluator with parameters from paper (Equations 1-2, 8)
        CT = params.get('CT', 0.2)  # Cost per unit time
        CE = params.get('CE', 1.34)  # Cost per unit energy
        delta_t = params.get('delta_t', 1)  # Time weight
        delta_e = params.get('delta_e', 1)  # Energy weight
        
        evaluator = UtilityEvaluator(CT=CT, CE=CE, delta_t=delta_t, delta_e=delta_e)

        print(f"\n{'='*60}")
        print(f"EVALUATOR: Searching for optimal offloading policy")
        print(f"  Tasks (N): {N}")
        print(f"  Locations: {n_locations} (0=IoT, 1-{n_locations-1}=Remote)")
        print(f"  Cost Model: U(w,p) = {delta_t}*T + {delta_e}*E")
        print(f"  CT={CT}, CE={CE}")
        print(f"{'='*60}\n")

        # Get LLM-guided candidate policies
        print("Using Chain-of-Thought to generate intelligent candidate policies...")
        llm_candidates = self.get_llm_guided_heuristics(workflow_dict, env_dict, plan, params)
        
        # Generate additional systematic candidates
        systematic_candidates = []
        
        # All local (baseline)
        systematic_candidates.append(tuple(0 for _ in range(N)))
        
        # All on first remote server
        if n_locations > 1:
            systematic_candidates.append(tuple(1 for _ in range(N)))
        
        # All on last server (typically cloud)
        if n_locations > 1:
            systematic_candidates.append(tuple(n_locations-1 for _ in range(N)))
        
        # Round-robin distribution
        if n_locations > 1:
            for start in range(min(n_locations, 3)):
                cand = tuple((start + i) % n_locations for i in range(N))
                systematic_candidates.append(cand)
        
        # Combine candidates
        candidates = llm_candidates + [c for c in systematic_candidates if c not in llm_candidates]
        
        # If problem is small, do exhaustive search
        max_exhaustive = 10000
        total_combos = n_locations ** N
        if total_combos <= max_exhaustive:
            print(f"✓ Problem size allows exhaustive search ({total_combos} combinations)")
            all_candidates = list(itertools.product(range(n_locations), repeat=N))
            candidates = list(set(candidates + all_candidates))
        else:
            print(f"⚠ Problem too large for exhaustive search ({total_combos} combinations)")
            print(f"  Using {len(candidates)} LLM-guided + heuristic candidates")

        print(f"\nEvaluating {len(candidates)} candidate policies using UtilityEvaluator...")
        print(f"  Computing U(w,p) via Equations 3-8 from paper\n")
        
        best_policy = None
        best_cost = float("inf")
        evaluated = 0
        skipped = 0

        for placement_tuple in candidates:
            try:
                # Convert to dict format: {1: loc1, 2: loc2, ..., N: locN}
                placement_dict = {i: placement_tuple[i-1] for i in range(1, N + 1)}
                
                # Use UtilityEvaluator to compute total offloading cost
                cost = evaluator.total_offloading_cost(workflow, placement_dict, env)
                evaluated += 1

                if cost is None or (isinstance(cost, float) and math.isinf(cost)):
                    skipped += 1
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_policy = placement_tuple
                    print(f"  ✓ New best: {best_policy} with U(w,p) = {best_cost:.6f}")

            except KeyError as e:
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
        workflow_dict = state.get("workflow")
        env_dict = state.get("env", {})
        params = state.get("params", {})
        plan = state.get("plan", "")

        if not isinstance(workflow_dict, dict):
            raise ValueError("workflow_dict must be a dictionary")

        result = self.find_best_policy(workflow_dict, env_dict, params, plan)

        if result["best_policy"] is None:
            evaluation = f"No finite-cost policy found. Evaluated={result['evaluated']}, Skipped={result['skipped']}"
            optimal_policy = []
        else:
            evaluation = f"Optimal policy found: U(w,p*) = {result['best_cost']:.6f}"
            optimal_policy = list(result['best_policy'])

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE:")
        print(f"  {evaluation}")
        print(f"  Evaluated: {result['evaluated']} policies")
        print(f"  Skipped: {result['skipped']} policies")
        print(f"{'='*60}\n")

        return {
            **state,
            "evaluation": evaluation,
            "optimal_policy": optimal_policy
        }