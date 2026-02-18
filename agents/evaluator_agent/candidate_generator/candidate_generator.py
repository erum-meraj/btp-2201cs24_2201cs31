"""
Candidate Policy Generator Tool

Generates candidate policies for the evaluator agent to explore.
Uses multiple strategies:
1. LLM-guided intelligent candidates
2. Systematic heuristics (all-local, all-edge, all-cloud)
3. Round-robin distribution
4. Memory-based similar policies
"""

import itertools
from typing import List, Tuple, Dict, Any, Set


class CandidatePolicyGenerator:
    """
    Generates candidate task offloading policies using multiple strategies.

    Strategies:
    - LLM-guided: Uses LLM reasoning to suggest promising policies
    - Systematic: All tasks to one location (local/edge/cloud)
    - Round-robin: Distribute tasks cyclically across locations
    - Memory-based: Retrieve similar past executions
    """

    def __init__(self, memory_manager=None):
        """
        Initialize the generator.

        Args:
            memory_manager: Optional memory system for retrieving similar cases
        """
        self.memory_manager = memory_manager

    def generate_candidates(
        self,
        num_tasks: int,
        location_ids: List[int],
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        llm_candidates: List[Tuple[int, ...]] = None,
        max_exhaustive: int = 10000,
    ) -> List[Tuple[int, ...]]:
        """
        Generate candidate policies using all available strategies.

        Args:
            num_tasks: Number of tasks in workflow
            location_ids: List of available location IDs
            workflow_dict: Workflow configuration
            env_dict: Environment configuration
            params: Cost parameters
            llm_candidates: Optional list of LLM-suggested policies
            max_exhaustive: Maximum combinations for exhaustive search

        Returns:
            List of candidate policy tuples
        """
        candidates = []

        # Strategy 1: LLM-guided candidates
        if llm_candidates:
            candidates.extend(llm_candidates)
            print(f"  ✓ Added {len(llm_candidates)} LLM-guided candidates")

        # Strategy 2: Memory-based candidates
        memory_candidates = self._get_memory_based_candidates(
            workflow_dict, env_dict, params, num_tasks
        )
        if memory_candidates:
            candidates.extend(memory_candidates)
            print(f"  ✓ Added {len(memory_candidates)} memory-based candidates")

        # Strategy 3: Systematic heuristics
        systematic = self._generate_systematic_candidates(num_tasks, location_ids)
        candidates.extend(systematic)
        print(f"  ✓ Added {len(systematic)} systematic heuristic candidates")

        # Strategy 4: Exhaustive search (if feasible)
        num_locations = len(location_ids)
        total_combos = num_locations**num_tasks

        if total_combos <= max_exhaustive:
            print(
                f"  ✓ Problem size allows exhaustive search ({total_combos} combinations)"
            )
            exhaustive = list(itertools.product(location_ids, repeat=num_tasks))
            candidates.extend(exhaustive)
        else:
            print(
                f"  ⚠  Problem too large for exhaustive ({total_combos} combinations)"
            )
            print(f"     Using heuristic-guided search instead")

        # Remove duplicates while preserving order
        return self._deduplicate(candidates)

    def _generate_systematic_candidates(
        self, num_tasks: int, location_ids: List[int]
    ) -> List[Tuple[int, ...]]:
        """
        Generate systematic heuristic candidates.

        Strategies:
        - All tasks to each location (all-local, all-edge, all-cloud)
        - Round-robin distribution across locations
        """
        candidates = []
        num_locations = len(location_ids)

        # All tasks to each location
        for loc_id in location_ids:
            policy = tuple(loc_id for _ in range(num_tasks))
            candidates.append(policy)

        # Round-robin patterns (up to 3 different starting points)
        for start_idx in range(min(num_locations, 3)):
            policy = tuple(
                location_ids[(start_idx + i) % num_locations] for i in range(num_tasks)
            )
            candidates.append(policy)

        return candidates

    def _get_memory_based_candidates(
        self,
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        num_tasks: int,
    ) -> List[Tuple[int, ...]]:
        """
        Retrieve candidate policies from similar past executions.

        Args:
            workflow_dict: Current workflow configuration
            env_dict: Current environment configuration
            params: Current parameters
            num_tasks: Number of tasks in current workflow

        Returns:
            List of policy tuples from memory
        """
        if not self.memory_manager:
            return []

        try:
            # Retrieve similar executions
            similar = self.memory_manager.retrieve_similar_executions(
                workflow_dict, env_dict, params, top_k=5
            )

            candidates = []
            for execution in similar:
                policy = execution.get("policy")
                if policy and len(policy) == num_tasks:
                    candidates.append(tuple(policy))

            return candidates

        except Exception as e:
            print(f"  ⚠  Error retrieving memory-based candidates: {e}")
            return []

    def _deduplicate(self, candidates: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Remove duplicate policies while preserving order."""
        seen: Set[Tuple[int, ...]] = set()
        unique = []

        for policy in candidates:
            if policy not in seen:
                seen.add(policy)
                unique.append(policy)

        return unique

    def filter_by_constraints(
        self,
        candidates: List[Tuple[int, ...]],
        fixed_locations: Dict[int, int] = None,
        allowed_locations: Dict[int, List[int]] = None,
    ) -> List[Tuple[int, ...]]:
        """
        Filter candidates by constraint satisfaction.

        Args:
            candidates: List of candidate policies
            fixed_locations: Dict of {task_id: required_location}
            allowed_locations: Dict of {task_id: [allowed_location_ids]}

        Returns:
            Filtered list of valid candidates
        """
        if not fixed_locations and not allowed_locations:
            return candidates

        valid_candidates = []

        for policy in candidates:
            if self._satisfies_constraints(policy, fixed_locations, allowed_locations):
                valid_candidates.append(policy)

        print(
            f"  ✓ Filtered to {len(valid_candidates)}/{len(candidates)} "
            f"constraint-satisfying candidates"
        )

        return valid_candidates

    def _satisfies_constraints(
        self,
        policy: Tuple[int, ...],
        fixed_locations: Dict[int, int] = None,
        allowed_locations: Dict[int, List[int]] = None,
    ) -> bool:
        """Check if a policy satisfies all constraints."""
        num_tasks = len(policy)

        # Check fixed location constraints
        if fixed_locations:
            for task_id, required_loc in fixed_locations.items():
                task_id = int(task_id)  # Normalize to int
                if 1 <= task_id <= num_tasks:
                    actual_loc = policy[task_id - 1]
                    if actual_loc != required_loc:
                        return False

        # Check allowed location constraints
        if allowed_locations:
            for task_id, allowed_locs in allowed_locations.items():
                task_id = int(task_id)  # Normalize to int
                if 1 <= task_id <= num_tasks:
                    actual_loc = policy[task_id - 1]
                    if actual_loc not in allowed_locs:
                        return False

        return True
