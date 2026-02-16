"""
Utility Function Tool

Wrapper around the cost evaluation functionality to provide
a clean interface for the evaluator agent.

This tool computes the offloading cost U(w,p) following
the paper's cost model (Equations 1-8).
"""

import math
from typing import Dict, Any, Tuple


class UtilityFunctionTool:
    """
    Tool for computing task offloading costs.
    
    Computes U(w,p) = delta_t * T + delta_e * E where:
    - T: Time consumption cost (Equation 7)
    - E: Energy consumption cost (Equation 3)
    """
    
    def __init__(self, evaluator):
        """
        Initialize the tool with a UtilityEvaluator instance.
        
        Args:
            evaluator: UtilityEvaluator instance from cost_eval module
        """
        self.evaluator = evaluator
    
    def evaluate_policy(
        self,
        policy_tuple: Tuple[int, ...],
        workflow,
        environment,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single policy and return detailed cost breakdown.
        
        Args:
            policy_tuple: Tuple of location IDs for each task
            workflow: Workflow object
            environment: Environment object
            verbose: If True, print detailed breakdown
            
        Returns:
            Dictionary with cost components:
            - total: Total offloading cost U(w,p)
            - time: Time consumption cost T
            - energy: Energy consumption cost E
            - ED: Data communication energy
            - EV: Task execution energy
            - delta_max: Critical path delay
        """
        num_tasks = workflow.N
        
        # Convert tuple to dict (1-indexed)
        policy_dict = {i: policy_tuple[i-1] for i in range(1, num_tasks + 1)}
        
        try:
            # Get detailed evaluation
            result = self.evaluator.evaluate(workflow, policy_dict, environment)
            
            if verbose:
                self._print_breakdown(policy_tuple, result)
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error evaluating policy {policy_tuple}: {e}")
            return {
                'total': float('inf'),
                'time': float('inf'),
                'energy': float('inf'),
                'ED': float('inf'),
                'EV': float('inf'),
                'delta_max': float('inf')
            }
    
    def evaluate_batch(
        self,
        candidates: list,
        workflow,
        environment,
        show_progress: bool = True
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """
        Evaluate a batch of candidate policies.
        
        Args:
            candidates: List of policy tuples
            workflow: Workflow object
            environment: Environment object
            show_progress: If True, print progress updates
            
        Returns:
            Dictionary mapping policy tuples to their cost breakdowns
        """
        results = {}
        total = len(candidates)
        
        for idx, policy in enumerate(candidates, 1):
            result = self.evaluate_policy(policy, workflow, environment, verbose=False)
            results[policy] = result
            
            # Show progress every 100 policies or at end
            if show_progress and (idx % 100 == 0 or idx == total):
                finite_count = sum(1 for r in results.values() if not math.isinf(r['total']))
                print(f"  Progress: {idx}/{total} policies evaluated "
                      f"({finite_count} finite-cost)")
        
        return results
    
    def find_best_policy(
        self,
        candidates: list,
        workflow,
        environment
    ) -> Dict[str, Any]:
        """
        Find the best policy from a list of candidates.
        
        Args:
            candidates: List of policy tuples
            workflow: Workflow object
            environment: Environment object
            
        Returns:
            Dictionary with:
            - best_policy: Optimal policy tuple
            - best_cost: Minimum cost achieved
            - evaluated: Number of policies evaluated
            - skipped: Number of policies with infinite cost
            - all_results: Dictionary of all evaluations
        """
        print(f"\nEvaluating {len(candidates)} candidate policies...")
        
        all_results = self.evaluate_batch(candidates, workflow, environment)
        
        best_policy = None
        best_cost = float('inf')
        evaluated = 0
        skipped = 0
        
        for policy, result in all_results.items():
            cost = result['total']
            
            if math.isinf(cost) or cost is None:
                skipped += 1
                continue
            
            evaluated += 1
            
            if cost < best_cost:
                best_cost = cost
                best_policy = policy
                print(f"  ✓ New best: {best_policy} with U(w,p) = {best_cost:.6f}")
        
        return {
            'best_policy': best_policy,
            'best_cost': best_cost,
            'evaluated': evaluated,
            'skipped': skipped,
            'all_results': all_results
        }
    
    def _print_breakdown(
        self,
        policy: Tuple[int, ...],
        result: Dict[str, float]
    ) -> None:
        """Print detailed cost breakdown for a policy."""
        print(f"\nPolicy: {policy}")
        print(f"  Total Cost U(w,p): {result['total']:.6f}")
        print(f"  Time Cost T: {result['time']:.6f}")
        print(f"  Energy Cost E: {result['energy']:.6f}")
        print(f"    ED (data comm): {result['ED']:.6f}")
        print(f"    EV (execution): {result['EV']:.6f}")
        print(f"  Critical Path Δ_max: {result['delta_max']:.6f} ms")