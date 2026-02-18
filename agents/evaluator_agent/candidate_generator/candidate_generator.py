"""
Candidate Policy Generator Tool - With Comprehensive Logging

Generates candidate policies with detailed logging of:
- Each generation strategy used
- Number of candidates from each source
- Constraint filtering
- Deduplication results
"""

import itertools
from typing import List, Tuple, Dict, Any, Set, Optional


class CandidatePolicyGenerator:
    """
    Generates candidate task offloading policies with detailed logging.
    """
    
    def __init__(self, memory_manager=None):
        """Initialize the generator."""
        self.memory_manager = memory_manager
        self.logger = None  # Will be set by caller
    
    def generate_candidates(
        self,
        num_tasks: int,
        location_ids: List[int],
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        llm_candidates: List[Tuple[int, ...]] = None,
        max_exhaustive: int = 10000,
        logger=None
    ) -> List[Tuple[int, ...]]:
        """
        Generate candidate policies with comprehensive logging.
        """
        self.logger = logger  # Store logger for use in sub-methods
        candidates = []
        
        if self.logger:
            self.logger.tool("CandGen", f"Starting generation: {num_tasks} tasks, {len(location_ids)} locations")
        
        # ========== Strategy 1: LLM-Guided Candidates ==========
        if llm_candidates:
            if self.logger:
                self.logger.tool_call(
                    "CandGen",
                    "add_llm_candidates",
                    {"count": len(llm_candidates), "preview": str(llm_candidates[:2])}
                )
            
            candidates.extend(llm_candidates)
            
            if self.logger:
                self.logger.tool_result(
                    "CandGen",
                    "add_llm_candidates",
                    f"Added {len(llm_candidates)} LLM-guided candidates"
                )
        
        # ========== Strategy 2: Memory-Based Candidates ==========
        if self.logger:
            self.logger.tool_call(
                "CandGen",
                "retrieve_memory_candidates",
                {"has_memory": bool(self.memory_manager), "top_k": 5}
            )
        
        memory_candidates = self._get_memory_based_candidates(
            workflow_dict, env_dict, params, num_tasks
        )
        
        if memory_candidates:
            candidates.extend(memory_candidates)
            if self.logger:
                self.logger.tool_result(
                    "CandGen",
                    "retrieve_memory_candidates",
                    f"Retrieved {len(memory_candidates)} from memory"
                )
        else:
            if self.logger:
                self.logger.tool_result(
                    "CandGen",
                    "retrieve_memory_candidates",
                    "No memory candidates found"
                )
        
        # ========== Strategy 3: Systematic Heuristics ==========
        if self.logger:
            self.logger.tool_call(
                "CandGen",
                "generate_systematic",
                {"num_tasks": num_tasks, "locations": location_ids}
            )
        
        systematic = self._generate_systematic_candidates(num_tasks, location_ids)
        candidates.extend(systematic)
        
        if self.logger:
            self.logger.tool_result(
                "CandGen",
                "generate_systematic",
                f"Generated {len(systematic)} systematic patterns (all-local, all-edge, round-robin)"
            )
        
        # ========== Strategy 4: Exhaustive Search ==========
        num_locations = len(location_ids)
        total_combos = num_locations ** num_tasks
        
        if self.logger:
            self.logger.tool_call(
                "CandGen",
                "check_exhaustive_feasibility",
                {
                    "total_combinations": total_combos,
                    "max_threshold": max_exhaustive,
                    "feasible": total_combos <= max_exhaustive
                }
            )
        
        if total_combos <= max_exhaustive:
            if self.logger:
                self.logger.tool("CandGen", f"Exhaustive search feasible: {total_combos} ≤ {max_exhaustive}")
            
            exhaustive = list(itertools.product(location_ids, repeat=num_tasks))
            candidates.extend(exhaustive)
            
            if self.logger:
                self.logger.tool_result(
                    "CandGen",
                    "generate_exhaustive",
                    f"Generated ALL {total_combos} combinations (complete search)"
                )
        else:
            if self.logger:
                self.logger.tool_result(
                    "CandGen",
                    "skip_exhaustive",
                    f"Too large: {total_combos:,} > {max_exhaustive:,} (using heuristics only)"
                )
        
        # ========== Deduplication ==========
        initial_count = len(candidates)
        
        if self.logger:
            self.logger.tool_call(
                "CandGen",
                "deduplicate",
                {"initial_count": initial_count}
            )
        
        unique_candidates = self._deduplicate(candidates)
        
        if self.logger:
            duplicates_removed = initial_count - len(unique_candidates)
            self.logger.tool_result(
                "CandGen",
                "deduplicate",
                f"Removed {duplicates_removed} duplicates → {len(unique_candidates)} unique"
            )
        
        return unique_candidates
    
    def _generate_systematic_candidates(
        self,
        num_tasks: int,
        location_ids: List[int]
    ) -> List[Tuple[int, ...]]:
        """Generate systematic heuristic candidates."""
        candidates = []
        num_locations = len(location_ids)
        
        # All tasks to each location
        for loc_id in location_ids:
            policy = tuple(loc_id for _ in range(num_tasks))
            candidates.append(policy)
        
        # Round-robin patterns
        for start_idx in range(min(num_locations, 3)):
            policy = tuple(
                location_ids[(start_idx + i) % num_locations]
                for i in range(num_tasks)
            )
            candidates.append(policy)
        
        return candidates
    
    def _get_memory_based_candidates(
        self,
        workflow_dict: Dict[str, Any],
        env_dict: Dict[str, Any],
        params: Dict[str, Any],
        num_tasks: int
    ) -> List[Tuple[int, ...]]:
        """Retrieve candidate policies from similar past executions."""
        if not self.memory_manager:
            return []
        
        try:
            similar = self.memory_manager.retrieve_similar_executions(
                workflow_dict, env_dict, params, top_k=5
            )
            
            candidates = []
            for execution in similar:
                policy = execution.get('policy')
                if policy and len(policy) == num_tasks:
                    candidates.append(tuple(policy))
            
            return candidates
            
        except Exception as e:
            if self.logger:
                self.logger.error("CandGen", e, context="Memory retrieval failed")
            return []
    
    def _deduplicate(
        self,
        candidates: List[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
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
        logger=None
    ) -> List[Tuple[int, ...]]:
        """
        Filter candidates by constraint satisfaction with logging.
        """
        self.logger = logger
        
        if not fixed_locations and not allowed_locations:
            return candidates
        
        if self.logger:
            self.logger.tool_call(
                "CandGen",
                "apply_constraints",
                {
                    "initial_count": len(candidates),
                    "fixed": fixed_locations,
                    "allowed": str(allowed_locations)[:100] if allowed_locations else None
                }
            )
        
        valid_candidates = []
        for policy in candidates:
            if self._satisfies_constraints(policy, fixed_locations, allowed_locations):
                valid_candidates.append(policy)
        
        filtered_count = len(candidates) - len(valid_candidates)
        
        if self.logger:
            self.logger.tool_result(
                "CandGen",
                "apply_constraints",
                f"Filtered {filtered_count} invalid → {len(valid_candidates)} valid candidates"
            )
        
        return valid_candidates
    
    def _satisfies_constraints(
        self,
        policy: Tuple[int, ...],
        fixed_locations: Dict[int, int] = None,
        allowed_locations: Dict[int, List[int]] = None
    ) -> bool:
        """Check if a policy satisfies all constraints."""
        num_tasks = len(policy)
        
        # Check fixed location constraints
        if fixed_locations:
            for task_id, required_loc in fixed_locations.items():
                task_id = int(task_id)
                if 1 <= task_id <= num_tasks:
                    actual_loc = policy[task_id - 1]
                    if actual_loc != required_loc:
                        return False
        
        # Check allowed location constraints
        if allowed_locations:
            for task_id, allowed_locs in allowed_locations.items():
                task_id = int(task_id)
                if 1 <= task_id <= num_tasks:
                    actual_loc = policy[task_id - 1]
                    if actual_loc not in allowed_locs:
                        return False
        
        return True