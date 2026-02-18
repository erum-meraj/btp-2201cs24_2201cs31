"""
Utility Function Tool - With Detailed Evaluation Logging

Logs every policy evaluation including:
- Individual policy costs
- Progress through batch evaluation
- Best policy updates
- Cost breakdowns
"""

import math
from typing import Dict, Any, Tuple, Optional


class UtilityFunctionTool:
    """
    Tool for computing task offloading costs with comprehensive logging.
    """
    
    def __init__(self, evaluator, logger=None):
        """
        Initialize the tool.
        
        Args:
            evaluator: UtilityEvaluator instance from cost_eval module
            logger: Optional AgenticLogger instance
        """
        self.evaluator = evaluator
        self.logger = logger
    
    def evaluate_policy(
        self,
        policy_tuple: Tuple[int, ...],
        workflow,
        environment,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single policy with optional logging.
        """
        num_tasks = workflow.N
        policy_dict = {i: policy_tuple[i-1] for i in range(1, num_tasks + 1)}
        
        try:
            result = self.evaluator.evaluate(workflow, policy_dict, environment)
            
            if verbose and self.logger:
                self.logger.tool(
                    "Utility",
                    f"Policy {policy_tuple} | Cost: {result['total']:.6f} "
                    f"(T: {result['time']:.6f}, E: {result['energy']:.6f})"
                )
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error("Utility", e, context=f"Evaluating policy {policy_tuple}")
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
        Evaluate a batch of candidate policies with progress logging.
        """
        results = {}
        total = len(candidates)
        
        if self.logger:
            self.logger.tool_call(
                "Utility",
                "evaluate_batch",
                {"num_candidates": total}
            )
        
        # Track progress milestones
        progress_points = [10, 25, 50, 75, 90, 100]
        next_progress_idx = 0
        
        for idx, policy in enumerate(candidates, 1):
            # Evaluate policy
            result = self.evaluate_policy(policy, workflow, environment, verbose=False)
            results[policy] = result
            
            # Calculate progress percentage
            progress_pct = (idx / total) * 100
            
            # Log at progress milestones
            if next_progress_idx < len(progress_points) and progress_pct >= progress_points[next_progress_idx]:
                if self.logger:
                    finite_count = sum(1 for r in results.values() if not math.isinf(r['total']))
                    self.logger.tool(
                        "Utility",
                        f"Progress: {idx}/{total} ({progress_pct:.0f}%) | "
                        f"{finite_count} valid policies found"
                    )
                next_progress_idx += 1
        
        if self.logger:
            finite_count = sum(1 for r in results.values() if not math.isinf(r['total']))
            self.logger.tool_result(
                "Utility",
                "evaluate_batch",
                f"Completed: {total} evaluated, {finite_count} valid, {total - finite_count} invalid"
            )
        
        return results
    
    def find_best_policy(
        self,
        candidates: list,
        workflow,
        environment
    ) -> Dict[str, Any]:
        """
        Find the best policy with detailed logging of the search process.
        """
        if self.logger:
            self.logger.tool_call(
                "Utility",
                "find_best_policy",
                {"num_candidates": len(candidates)}
            )
        
        # Evaluate all candidates
        all_results = self.evaluate_batch(candidates, workflow, environment)
        
        best_policy = None
        best_cost = float('inf')
        evaluated = 0
        skipped = 0
        
        # Track improvement events
        improvement_count = 0
        
        for policy, result in all_results.items():
            cost = result['total']
            
            if math.isinf(cost) or cost is None:
                skipped += 1
                continue
            
            evaluated += 1
            
            # Check if this is a new best
            if cost < best_cost:
                improvement_count += 1
                old_best = best_cost
                best_cost = cost
                best_policy = policy
                
                # Log each improvement
                if self.logger:
                    if old_best == float('inf'):
                        self.logger.policy_evaluation(policy, cost, is_best=True)
                    else:
                        improvement = ((old_best - cost) / old_best) * 100
                        self.logger.tool(
                            "Utility",
                            f"Improvement #{improvement_count}: {policy} | "
                            f"Cost: {cost:.6f} (↓ {improvement:.1f}%)"
                        )
                        self.logger.policy_evaluation(policy, cost, is_best=True)
        
        # Final summary
        if self.logger:
            if best_policy:
                self.logger.tool_result(
                    "Utility",
                    "find_best_policy",
                    f"Optimal: {best_policy} | Cost: {best_cost:.6f} | "
                    f"Improvements: {improvement_count} | "
                    f"Stats: {evaluated} valid, {skipped} invalid"
                )
            else:
                self.logger.tool_result(
                    "Utility",
                    "find_best_policy",
                    f"No valid policy found | {evaluated} evaluated, {skipped} skipped"
                )
        
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
        """Print detailed cost breakdown (kept for backward compatibility)."""
        if self.logger:
            self.logger.tool(
                "Utility",
                f"Breakdown for {policy}: "
                f"Total={result['total']:.6f}, "
                f"Time={result['time']:.6f}, "
                f"Energy={result['energy']:.6f}, "
                f"ED={result['ED']:.6f}, "
                f"EV={result['EV']:.6f}, "
                f"Δ_max={result['delta_max']:.6f}ms"
            )