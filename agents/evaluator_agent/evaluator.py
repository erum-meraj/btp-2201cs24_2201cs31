"""
Evaluator Agent - Refactored with Tool-Based Architecture

The evaluator agent orchestrates policy search using:
1. Candidate Policy Generator - generates promising policies
2. Utility Function Tool - evaluates policy costs
3. Weak Solver - advanced optimization (placeholder)
4. Memory - retrieves similar past executions
"""

import os
import json
from typing import Dict, Any, List, Tuple
from agents.evaluator_agent.candidate_generator.candidate_generator import (
    CandidateGenerationAgent,
)
from agents.evaluator_agent.tools.utility_function import UtilityFunctionTool
from agents.evaluator_agent.weak_solver.weak_solver import WeakSolverTool
from core.logger import get_logger

class EvaluatorAgent:
    """
    Supervisor agent that coordinates policy search.

    Architecture:
    - Uses tools (candidate generator, utility evaluator, weak solver)
    - Employs LLM for intelligent heuristic guidance
    - Delegates actual optimization to specialized tools
    """

    def __init__(
        self,
        base_agent,
        workflow_module,
        environment_module,
        cost_evaluator_class,
        memory_manager=None,
        log_file: str = "agent_trace.txt",
    ):
        """
        Initialize the evaluator agent.

        Args:
            base_agent: BaseAgent instance for LLM interactions
            workflow_module: Workflow class from core.workflow
            environment_module: Environment class from core.environment
            cost_evaluator_class: UtilityEvaluator class from core.cost_eval
            memory_manager: Optional memory system
            log_file: Path to interaction log file
        """
        self.base_agent = base_agent
        self.Workflow = workflow_module
        self.Environment = environment_module
        self.UtilityEvaluator = cost_evaluator_class
        self.memory_manager = memory_manager
        self.log_file = log_file

        # Initialize tools
        self.candidate_generator = CandidateGenerationAgent(base_agent, memory_manager=self.memory_manager)
        self.weak_solver = WeakSolverTool()

        # Utility tool will be initialized per run (needs evaluator instance)
        self.utility_tool = None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the evaluator agent workflow.

        Args:
            state: Current state with env, workflow, params, plan

        Returns:
            Updated state with evaluation, optimal_policy, best_cost
        """
        try:
            
            logger = get_logger(self.log_file)
            logger.evaluator("Starting policy search and evaluation")
        except ImportError:
            logger = None
            
        # Extract state components
        workflow_dict = state.get("workflow")
        env_dict = state.get("env", {})
        params = state.get("params", {})
        plan = state.get("plan", "")

        # Validate input
        if not isinstance(workflow_dict, dict):
            raise ValueError("workflow_dict must be a dictionary")

        # Find optimal policy
        result = self.find_best_policy(workflow_dict, env_dict, params, plan)

        # Format results
        if result["best_policy"] is None:
            evaluation = (
                f"No finite-cost policy found. "
                f"Evaluated={result['evaluated']}, "
                f"Skipped={result['skipped']}"
            )
            optimal_policy = []
            best_cost = float("inf")
            if logger:
                logger.warning("Evaluator", evaluation)
        else:
            evaluation = f"Optimal policy found: U(w,p*) = {result['best_cost']:.6f}"
            optimal_policy = list(result["best_policy"])
            best_cost = result["best_cost"]
            if logger:
                logger.evaluator(f"Optimal policy: {optimal_policy} | Cost: {best_cost:.6f}")
                logger.evaluator(f"Search complete: {result['evaluated']} evaluated, {result['skipped']} skipped")

        return {
            **state,
            "evaluation": evaluation,
            "optimal_policy": optimal_policy,
            "best_cost": best_cost,
        }

    def find_best_policy(
        self,
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        plan: str = "",
    ) -> Dict[str, Any]:
        """
        Search for optimal placement policy.

        Process:
        1. Get LLM-guided heuristic suggestions
        2. Generate candidate policies using multiple strategies
        3. Evaluate all candidates with utility function
        4. Optionally apply weak solver for refinement

        Args:
            workflow_dict: Workflow configuration
            env_dict: Environment configuration
            params: Cost model parameters
            plan: Strategic plan from planner agent

        Returns:
            Dictionary with best_policy, best_cost, evaluated, skipped
        """
        # Create workflow and environment objects
        workflow = self.Workflow.from_experiment_dict(workflow_dict)
        env = self._create_environment(env_dict)

        num_tasks = workflow.N
        location_ids = sorted(env_dict.get("locations", {}).keys())
        num_locations = len(location_ids)

        # Initialize utility evaluator and tool
        evaluator = self.UtilityEvaluator(
            CT=params.get("CT", 0.2),
            CE=params.get("CE", 1.34),
            delta_t=params.get("delta_t", 1),
            delta_e=params.get("delta_e", 1),
        )
        self.utility_tool = UtilityFunctionTool(evaluator)

        # Display problem info
        self._print_problem_info(num_tasks, num_locations, location_ids, params)

        # Step 1: Get LLM-guided candidates
        print("\n[Step 1/3] Generating LLM-guided candidate policies...")
        llm_candidates = self._get_llm_candidates(
            workflow_dict, env_dict, plan, params, location_ids
        )

        # Step 2: Generate all candidates
        print("\n[Step 2/3] Generating candidate policies from all sources...")
        # Use candidate_generator's generate_candidates when available; otherwise fall back
        if hasattr(self.candidate_generator, "generate_candidates"):
            candidates = self.candidate_generator.generate_candidates(
                num_tasks=num_tasks,
                location_ids=location_ids,
                workflow_dict=workflow_dict,
                env_dict=env_dict,
                params=params,
                llm_candidates=llm_candidates,
                max_exhaustive=10000,
            )
        else:
            # Backwards-compatible fallback: try to use generate_next_batch or simple systematic generation
            candidates = []
            if llm_candidates:
                candidates.extend(llm_candidates)

            try:
                # Ask the older agent for several batches
                batch = self.candidate_generator.generate_next_batch(
                    num_tasks, location_ids, workflow_dict, env_dict, params, batch_size=50
                )
                candidates.extend(batch)
            except Exception:
                # Systematic fallbacks
                from itertools import product
                for loc in location_ids:
                    candidates.append(tuple(loc for _ in range(num_tasks)))
                for start in range(min(len(location_ids), 3)):
                    candidates.append(tuple(location_ids[(start + i) % len(location_ids)] for i in range(num_tasks)))

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for c in candidates:
                t = tuple(int(x) for x in c)
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            candidates = unique

        # Apply constraints
        fixed = params.get("fixed_locations", {})
        allowed = params.get("allowed_locations", None)
        if fixed or allowed:
            candidates = self.candidate_generator.filter_by_constraints(
                candidates, fixed, allowed
            )

        # Step 3: Evaluate candidates
        print(
            f"\n[Step 3/3] Evaluating {len(candidates)} candidates using utility function..."
        )
        result = self.utility_tool.find_best_policy(candidates, workflow, env)

        # Optional: Apply weak solver
        if self.weak_solver.is_enabled() and result["best_policy"]:
            print("\n[Optional] Applying weak solver for refinement...")
            refined = self.weak_solver.solve(
                num_tasks=num_tasks,
                location_ids=location_ids,
                workflow=workflow,
                environment=env,
                initial_policy=result["best_policy"],
            )
            if refined["cost"] < result["best_cost"]:
                print(
                    f"  ✓ Weak solver improved cost: "
                    f"{result['best_cost']:.6f} → {refined['cost']:.6f}"
                )
                result["best_policy"] = refined["policy"]
                result["best_cost"] = refined["cost"]

        return result

    def _get_llm_candidates(
        self,
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        plan: str,
        params: Dict[str, Any],
        location_ids: List[int],
    ) -> List[Tuple[int, ...]]:
        """
        Use LLM with Chain-of-Thought to generate intelligent candidates.

        Args:
            workflow_dict: Workflow configuration
            env_dict: Environment configuration
            plan: Strategic plan from planner
            params: Parameters
            location_ids: Available location IDs

        Returns:
            List of promising policy tuples
        """
        tasks = workflow_dict.get("tasks", {})
        edges = workflow_dict.get("edges", {})
        num_tasks = workflow_dict.get("N", 0)

        # Format environment details
        env_details = self._format_env_for_prompt(env_dict)

        # Format task details
        task_details = self._format_task_details(tasks, edges)

        # Load prompt template - require prompt.md to be present in evaluator folder
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.md")
        if not os.path.exists(prompt_file_path):
            msg = f"Required prompt template not found: {prompt_file_path}"
            try:
                logger = get_logger(self.log_file)
                logger.error("Evaluator", msg)
            except Exception:
                pass
            raise FileNotFoundError(msg)

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(
            env_details=env_details,
            N=num_tasks,
            task_details=task_details,
            params=json.dumps(params, indent=2),
            plan=(plan or "")[:800],
            location_ids=location_ids,
        )

        # Log and get LLM response
        self._log_interaction("EVALUATOR", prompt, None, "PROMPT")
        result = self.base_agent.think_with_cot(prompt, return_reasoning=True)

        full_response = (
            f"REASONING:\n{result['reasoning']}\n\nCANDIDATES:\n{result['answer']}"
        )
        self._log_interaction("EVALUATOR", None, full_response, "RESPONSE")

        print(f"  ✓ LLM reasoning complete")

        # Parse policies from text
        policies = self._parse_policies_from_text(
            result["answer"], num_tasks, location_ids
        )

        return policies

    def _format_task_details(self, tasks: Dict, edges) -> str:
        """Format task details for prompt."""
        # Convert edges list to dict if necessary
        edges_dict = {}
        if isinstance(edges, list):
            # Format: [{u: int, v: int, bytes: float}, ...]
            for edge in edges:
                u = int(edge.get("u"))
                v = int(edge.get("v"))
                bytes_val = float(edge.get("bytes"))
                edges_dict[(u, v)] = bytes_val
        else:
            edges_dict = edges

        details = []
        for task_id in sorted(tasks.keys()):
            task_data = tasks[task_id]
            cpu_cycles = task_data.get("v", 0)
            parents = [j for (j, k), _ in edges_dict.items() if k == int(task_id)]
            children_deps = [
                (k, edges_dict[(int(task_id), k)])
                for (j, k), _ in edges_dict.items()
                if j == int(task_id)
            ]

            details.append(f"\nTask {task_id}:")
            details.append(f"  v_{task_id} = {cpu_cycles:.2e} CPU cycles")
            if parents:
                details.append(f"  Parents: {parents}")
            if children_deps:
                details.append(f"  Children:")
                for k, data_size in children_deps:
                    details.append(f"    Task {k}: {data_size:.2e} bytes")

        return "\n".join(details)

    def _format_env_for_prompt(self, env_dict: Dict[str, Any]) -> str:
        """Format environment details for LLM prompt."""
        details = []

        locations = env_dict.get("locations", {})
        if locations:
            details.append("Locations:")
            for loc_id, loc_type in sorted(locations.items()):
                details.append(f"  {loc_id}: {loc_type.upper()}")
            details.append("")

        # Add abbreviated info for other parameters
        if "DR" in env_dict:
            details.append(
                f"Data transfer rates available: {len(env_dict['DR'])} pairs"
            )
        if "VR" in env_dict:
            details.append(f"Computation speeds: {len(env_dict['VR'])} locations")

        return "\n".join(details)

    def _parse_policies_from_text(
        self, text: str, num_tasks: int, valid_location_ids: List[int]
    ) -> List[Tuple[int, ...]]:
        """Extract policy suggestions from LLM text output."""
        import re

        valid_set = set(valid_location_ids)
        policies = []

        pattern = r"[\[\(](\d+(?:\s*,\s*\d+)*)[\]\)]"
        matches = re.findall(pattern, text or "")

        for match in matches:
            try:
                policy = [int(x.strip()) for x in match.split(",")]
                if len(policy) == num_tasks and all(loc in valid_set for loc in policy):
                    policies.append(tuple(policy))
            except:
                continue

        # Remove duplicates
        unique = []
        seen = set()
        for p in policies:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique[:5]  # Return top 5

    def _create_environment(self, env_dict: Dict[str, Any]):
        """Create Environment object from dictionary."""
        return self.Environment.from_matrices(
            types=env_dict.get("locations", {}),
            DR_matrix=env_dict.get("DR", {}),
            DE_vector=env_dict.get("DE", {}),
            VR_vector=env_dict.get("VR", {}),
            VE_vector=env_dict.get("VE", {}),
        )

    def _print_problem_info(
        self,
        num_tasks: int,
        num_locations: int,
        location_ids: List[int],
        params: Dict[str, Any],
    ) -> None:
        """Print problem information."""
        print(f"\n{'='*60}")
        print(f"EVALUATOR: Searching for optimal offloading policy")
        print(f"  Tasks (N): {num_tasks}")
        print(f"  Locations: {num_locations} with IDs {location_ids}")
        print(
            f"  Cost Model: U(w,p) = {params.get('delta_t', 1)}*T + "
            f"{params.get('delta_e', 1)}*E"
        )
        print(f"  CT={params.get('CT', 0.2)}, CE={params.get('CE', 1.34)}")

        fixed = params.get("fixed_locations", {})
        allowed = params.get("allowed_locations", None)
        if fixed:
            print(f"  Fixed Constraints: {fixed}")
        if allowed:
            print(f"  Allowed Constraints: {allowed}")

        print(f"{'='*60}\n")

    def _log_interaction(
        self, agent: str, prompt: str, response: str, msg_type: str
    ) -> None:
        """Log agent interactions to file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            if msg_type == "PROMPT":
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{agent} AGENT - PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(prompt)
                f.write("\n" + "=" * 80 + "\n\n")
            elif msg_type == "RESPONSE":
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{agent} AGENT - RESPONSE\n")
                f.write("=" * 80 + "\n")
                f.write(response)
                f.write("\n" + "=" * 80 + "\n\n")