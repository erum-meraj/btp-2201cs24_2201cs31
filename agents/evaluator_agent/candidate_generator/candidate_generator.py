"""
Candidate Generation Agent

Generates promising candidate policies using LLM reasoning:
- Analyzes problem structure
- Learns from evaluation results  
- Refines search iteratively
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import json


class CandidateGenerationAgent:
    """
    Generates candidate policies using LLM reasoning and learning.
    
    Uses iterative approach:
    - Analyzes problem and generates promising candidates
    - Learns from evaluation results
    - Refines search until convergence
    """
    
    def __init__(self, base_agent, memory_manager=None, logger=None):
        """
        Initialize the intelligent candidate agent.
        
        Args:
            base_agent: BaseAgent for LLM access
            memory_manager: Optional memory for historical learning
            logger: AgenticLogger instance
        """
        self.base_agent = base_agent
        self.memory_manager = memory_manager
        self.logger = logger
        
        # Search state
        self.explored_policies = set()
        self.evaluation_history = []
        self.best_cost = float('inf')
        self.best_policy = None
        self.iteration = 0
    
    def generate_next_batch(
        self,
        num_tasks: int,
        location_ids: List[int],
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        batch_size: int = 10,
        utility_tool: Optional[Any] = None
    ) -> List[Tuple[int, ...]]:
        """
        Use LLM reasoning to generate next batch of promising candidates.
        
        This is the core intelligent loop:
        1. LLM analyzes problem + evaluation history
        2. LLM proposes promising policies to explore
        3. Return candidates for evaluation
        
        Args:
            num_tasks: Number of tasks
            location_ids: Available locations
            workflow_dict: Workflow structure
            env_dict: Environment parameters
            params: Cost parameters
            batch_size: Number of candidates to generate
            utility_tool: Reference to utility evaluator (for validation)
            
        Returns:
            List of policy tuples to evaluate next
        """
        self.iteration += 1
        
        if self.logger:
            self.logger.tool_call(
                "CandidateAgent",
                "generate_next_batch",
                {
                    "iteration": self.iteration,
                    "batch_size": batch_size,
                    "explored": len(self.explored_policies),
                    "best_cost": self.best_cost if self.best_cost != float('inf') else None
                }
            )
        
        # Build reasoning prompt for LLM
        prompt = self._build_reasoning_prompt(
            num_tasks, location_ids, workflow_dict, env_dict, params, batch_size
        )
        
        if self.logger:
            self.logger.llm_call("CandidateAgent", prompt[:300])
        
        # Get LLM's intelligent suggestions
        try:
            response = self.base_agent.think_with_cot(prompt, return_reasoning=True)
            reasoning = response.get('reasoning', '')
            answer = response.get('answer', '')
            
            if self.logger:
                self.logger.llm_response("CandidateAgent", f"Reasoning: {reasoning[:200]}...")
            
            # Parse candidates from LLM response
            candidates = self._parse_candidates(answer, num_tasks, location_ids)
            
            # Add to explored set
            for candidate in candidates:
                self.explored_policies.add(candidate)
            
            if self.logger:
                self.logger.tool_result(
                    "CandidateAgent",
                    "generate_next_batch",
                    f"Generated {len(candidates)} intelligent candidates based on reasoning"
                )
            
            return candidates
            
        except Exception as e:
            if self.logger:
                self.logger.error("CandidateAgent", e, context="LLM candidate generation")
            
            # Fallback to systematic exploration
            return self._fallback_candidates(num_tasks, location_ids, batch_size)
    
    def _build_reasoning_prompt(
        self,
        num_tasks: int,
        location_ids: List[int],
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        batch_size: int
    ) -> str:
        """
        Build intelligent reasoning prompt for LLM.
        
        The prompt includes:
        - Problem structure (tasks, dependencies, locations)
        - Evaluation history (what worked, what didn't)
        - Search progress (explored space, improvements)
        - Request for next promising candidates
        """
        # Format problem structure
        problem_summary = self._format_problem(workflow_dict, env_dict, params, num_tasks, location_ids)

        # Format evaluation history
        history_summary = self._format_evaluation_history()

        # Format search state
        search_state = self._format_search_state(num_tasks, location_ids)

        # Load prompt template file (prompt.md) from same folder and require it to exist
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.md")
        if not os.path.exists(prompt_file_path):
            msg = f"Required prompt template not found: {prompt_file_path}"
            if self.logger:
                self.logger.error("CandidateAgent", msg)
            raise FileNotFoundError(msg)

        # Read and format the template
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template = f.read()

        try:
            prompt = template.format(
                batch_size=batch_size,
                problem_summary=problem_summary,
                history_summary=history_summary,
                search_state=search_state,
            )
        except Exception as e:
            msg = f"Failed to format prompt template: {e}"
            if self.logger:
                self.logger.error("CandidateAgent", msg)
            raise

        return prompt
    
    def _format_problem(
        self,
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        num_tasks: int,
        location_ids: List[int]
    ) -> str:
        """Format problem structure for LLM."""
        tasks = workflow_dict.get('tasks', {})
        edges = workflow_dict.get('edges', {})
        
        lines = []
        lines.append(f"- Tasks: {num_tasks}")
        lines.append(f"- Locations: {location_ids} (0=IoT, 1=Edge, 2=Cloud)")
        lines.append(f"- Mode: delta_t={params.get('delta_t')}, delta_e={params.get('delta_e')}")
        lines.append("")
        
        # Task characteristics
        lines.append("Task Characteristics:")
        for task_id, task_data in sorted(tasks.items()):
            v = task_data.get('v', 0)
            lines.append(f"  Task {task_id}: {v:.2e} CPU cycles")
        lines.append("")
        
        # Key dependencies
        lines.append("Key Dependencies (high data transfer):")
        if isinstance(edges, dict):
            high_transfer = [(u, v, size) for (u, v), size in edges.items() 
                           if size > 1e6 and u > 0 and v <= num_tasks + 1]
            for u, v, size in sorted(high_transfer, key=lambda x: -x[2])[:5]:
                lines.append(f"  Task {u} → Task {v}: {size:.2e} bytes")
        
        return "\n".join(lines)
    
    def _format_evaluation_history(self) -> str:
        """Format evaluation history for LLM learning."""
        if not self.evaluation_history:
            return "No evaluations yet (first iteration)."
        
        lines = []
        lines.append(f"Evaluated {len(self.evaluation_history)} policies so far.")
        lines.append("")
        
        # Show best policies
        sorted_history = sorted(self.evaluation_history, key=lambda x: x['cost'])[:5]
        lines.append("Top 5 Best Policies Found:")
        for i, entry in enumerate(sorted_history, 1):
            lines.append(f"  {i}. Policy {entry['policy']} → Cost: {entry['cost']:.6f}")
        lines.append("")
        
        # Show worst policies (to avoid)
        worst = sorted(self.evaluation_history, key=lambda x: -x['cost'] if x['cost'] != float('inf') else 0)[:3]
        if worst:
            lines.append("Worst Policies (avoid similar):")
            for entry in worst:
                if entry['cost'] != float('inf'):
                    lines.append(f"  - Policy {entry['policy']} → Cost: {entry['cost']:.6f}")
        
        return "\n".join(lines)
    
    def _format_search_state(self, num_tasks: int, location_ids: List[int]) -> str:
        """Format current search state."""
        total_space = len(location_ids) ** num_tasks
        explored_pct = (len(self.explored_policies) / total_space) * 100 if total_space > 0 else 0
        
        lines = []
        lines.append(f"- Total search space: {total_space:,} possible policies")
        lines.append(f"- Explored so far: {len(self.explored_policies)} ({explored_pct:.2f}%)")
        lines.append(f"- Current best: Cost = {self.best_cost:.6f}" if self.best_cost != float('inf') else "- Current best: None found yet")
        lines.append(f"- Iteration: {self.iteration}")
        
        return "\n".join(lines)
    
    def _parse_candidates(
        self,
        llm_response: str,
        num_tasks: int,
        location_ids: List[int]
    ) -> List[Tuple[int, ...]]:
        """Parse LLM response to extract candidate policies."""
        candidates = []
        
        try:
            # Try to parse as JSON
            if '```json' in llm_response:
                json_str = llm_response.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_response:
                json_str = llm_response.split('```')[1].split('```')[0].strip()
            else:
                json_str = llm_response
            
            data = json.loads(json_str)
            
            # Extract candidates
            for candidate_obj in data.get('candidates', []):
                policy = candidate_obj.get('policy', [])
                if len(policy) == num_tasks:
                    policy_tuple = tuple(policy)
                    # Only add if not already explored
                    if policy_tuple not in self.explored_policies:
                        candidates.append(policy_tuple)
            
            if self.logger and candidates:
                self.logger.tool("CandidateAgent", f"Parsed {len(candidates)} valid candidates from LLM")
            
        except Exception as e:
            if self.logger:
                self.logger.warning("CandidateAgent", f"Failed to parse LLM response: {e}")
        
        return candidates
    
    def _fallback_candidates(
        self,
        num_tasks: int,
        location_ids: List[int],
        batch_size: int
    ) -> List[Tuple[int, ...]]:
        """Fallback to systematic exploration if LLM fails."""
        candidates = []
        
        # All-local, all-edge, all-cloud
        for loc in location_ids:
            policy = tuple(loc for _ in range(num_tasks))
            if policy not in self.explored_policies:
                candidates.append(policy)
                self.explored_policies.add(policy)
        
        # Round-robin patterns
        for start in range(min(len(location_ids), 3)):
            policy = tuple(location_ids[(start + i) % len(location_ids)] for i in range(num_tasks))
            if policy not in self.explored_policies:
                candidates.append(policy)
                self.explored_policies.add(policy)
        
        if self.logger:
            self.logger.tool("CandidateAgent", f"Fallback: Generated {len(candidates)} systematic candidates")
        
        return candidates[:batch_size]
    
    def update_from_evaluation(
        self,
        policy: Tuple[int, ...],
        cost: float,
        breakdown: Dict[str, float]
    ):
        """
        Update agent's understanding based on evaluation results.
        
        This creates a learning loop where the agent improves its
        candidate generation based on what works and what doesn't.
        """
        if self.logger:
            self.logger.tool_call(
                "CandidateAgent",
                "update_from_evaluation",
                {"policy": policy, "cost": cost}
            )
        
        # Add to history
        self.evaluation_history.append({
            'policy': policy,
            'cost': cost,
            'breakdown': breakdown,
            'iteration': self.iteration
        })
        
        # Update best
        if cost < self.best_cost:
            old_best = self.best_cost
            self.best_cost = cost
            self.best_policy = policy
            
            if self.logger:
                improvement = ((old_best - cost) / old_best * 100) if old_best != float('inf') else 100
                self.logger.tool_result(
                    "CandidateAgent",
                    "update_from_evaluation",
                    f"New best found! Cost: {cost:.6f} (↓ {improvement:.1f}%)"
                )
        else:
            if self.logger:
                self.logger.tool_result(
                    "CandidateAgent",
                    "update_from_evaluation",
                    f"Policy evaluated: {cost:.6f} (current best: {self.best_cost:.6f})"
                )
    
    def should_continue(self, max_iterations: int = 10, convergence_threshold: int = 3) -> bool:
        """
        Determine if search should continue.
        
        Args:
            max_iterations: Maximum iterations to run
            convergence_threshold: Stop if no improvement for N iterations
            
        Returns:
            True if should continue, False if converged or max reached
        """
        if self.iteration >= max_iterations:
            if self.logger:
                self.logger.tool("CandidateAgent", f"Reached max iterations ({max_iterations})")
            return False
        
        # Check for convergence (no improvement in last N iterations)
        if len(self.evaluation_history) >= convergence_threshold:
            recent = self.evaluation_history[-convergence_threshold:]
            best_in_recent = min(r['cost'] for r in recent)
            
            if best_in_recent >= self.best_cost:
                if self.logger:
                    self.logger.tool(
                        "CandidateAgent",
                        f"Converged: No improvement in last {convergence_threshold} iterations"
                    )
                return False
        
        return True
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search process."""
        return {
            'total_iterations': self.iteration,
            'total_evaluated': len(self.evaluation_history),
            'best_policy': self.best_policy,
            'best_cost': self.best_cost,
            'explored_percentage': (len(self.explored_policies) / (3 ** 4)) * 100  # Assuming 4 tasks, 3 locs
        }

    def generate_candidates(
        self,
        num_tasks: int,
        location_ids: List[int],
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        llm_candidates: Optional[List[Tuple[int, ...]]] = None,
        max_exhaustive: int = 10000,
    ) -> List[Tuple[int, ...]]:
        """
        Generate a list of candidate policies combining LLM suggestions and systematic exploration.

        Returns a de-duplicated list of policy tuples (length = num_tasks).
        """
        from itertools import product

        candidates: List[Tuple[int, ...]] = []
        seen = set()

        # Helper to add candidate
        def add(policy_tuple):
            if policy_tuple not in seen:
                seen.add(policy_tuple)
                candidates.append(policy_tuple)

        # 1) Add LLM-provided candidates first (if any)
        if llm_candidates:
            for p in llm_candidates:
                try:
                    pt = tuple(int(x) for x in p)
                except Exception:
                    continue
                if len(pt) == num_tasks:
                    add(pt)

        # 2) Add skyline/simple patterns: all-local/all-edge/all-cloud and round-robin
        for loc in location_ids:
            policy = tuple(loc for _ in range(num_tasks))
            add(policy)

        for start in range(min(len(location_ids), num_tasks)):
            policy = tuple(location_ids[(start + i) % len(location_ids)] for i in range(num_tasks))
            add(policy)

        # 3) Small perturbations around best_policy
        if self.best_policy:
            for i in range(num_tasks):
                for loc in location_ids:
                    if loc != self.best_policy[i]:
                        p = list(self.best_policy)
                        p[i] = loc
                        add(tuple(p))

        # 4) If still under the requested budget, sample exhaustively (bounded)
        total_possible = len(location_ids) ** num_tasks
        if len(candidates) < min(max_exhaustive, total_possible):
            limit = min(max_exhaustive, total_possible)
            count = len(candidates)
            for combo in product(location_ids, repeat=num_tasks):
                if combo in seen:
                    continue
                add(combo)
                count += 1
                if count >= limit:
                    break

        # Record explored policies
        for c in candidates:
            self.explored_policies.add(c)

        if self.logger:
            self.logger.tool_result("CandidateAgent", f"generate_candidates produced {len(candidates)} candidates")

        return candidates

    def filter_by_constraints(
        self,
        candidates: List[Tuple[int, ...]],
        fixed: Dict[Any, Any],
        allowed: Optional[Dict[Any, List[int]]] = None,
    ) -> List[Tuple[int, ...]]:
        """
        Filter candidate policies by fixed and allowed constraints.

        - fixed: mapping task_index -> required_location (task indices are 1-based or 0-based; we accept both)
        - allowed: mapping task_index -> list_of_allowed_locations
        """
        filtered = []

        def task_index_to_pos(k):
            try:
                ki = int(k)
            except Exception:
                return None
            # Accept 1-based task indices (common in this repo) by converting to 0-based if necessary
            if 1 <= ki <= (candidates and len(candidates[0]) or 0):
                return ki - 1
            # Otherwise assume it's already 0-based
            return ki

        for policy in candidates:
            ok = True
            # Check fixed
            for tk, loc in (fixed or {}).items():
                pos = task_index_to_pos(tk)
                if pos is None or pos < 0 or pos >= len(policy):
                    ok = False
                    break
                if policy[pos] != int(loc):
                    ok = False
                    break

            if not ok:
                continue

            # Check allowed
            if allowed:
                for tk, allowed_locs in allowed.items():
                    pos = task_index_to_pos(tk)
                    if pos is None or pos < 0 or pos >= len(policy):
                        ok = False
                        break
                    if policy[pos] not in allowed_locs:
                        ok = False
                        break

            if ok:
                filtered.append(policy)

        if self.logger:
            self.logger.tool_result("CandidateAgent", f"filter_by_constraints reduced {len(candidates)}→{len(filtered)} candidates")

        return filtered