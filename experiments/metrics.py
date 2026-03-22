"""
Metrics and Evaluation

Wrapper for evaluating policies and collecting statistics.
"""

import time
import math
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict


class MetricsEvaluator:
    """Evaluates policies and collects performance metrics."""
    
    def __init__(self, utility_evaluator_class, workflow_class, environment_class):
        self.UtilityEvaluator = utility_evaluator_class
        self.Workflow = workflow_class
        self.Environment = environment_class
    
    def evaluate_policy(self, policy: List[int], workflow_dict: Dict[str, Any], env_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single policy and return all metrics."""
        start_time = time.time()
        
        try:
            workflow = self.Workflow.from_experiment_dict(workflow_dict)
            env = self.Environment.from_matrices(
                types=env_dict.get('locations', {}),
                DR_matrix=env_dict.get('DR', {}),
                DE_vector=env_dict.get('DE', {}),
                VR_vector=env_dict.get('VR', {}),
                VE_vector=env_dict.get('VE', {})
            )
            
            evaluator = self.UtilityEvaluator(
                CT=params.get('CT', 0.2),
                CE=params.get('CE', 1.34),
                delta_t=params.get('delta_t', 1),
                delta_e=params.get('delta_e', 1)
            )
            
            num_tasks = workflow.N
            policy_dict = {i + 1: policy[i] for i in range(num_tasks)}
            result = evaluator.evaluate(workflow, policy_dict, env)
            end_time = time.time()
            
            return {
                'cost': result['total'],
                'time': result['time'],
                'energy': result['energy'],
                'ED': result['ED'],
                'EV': result['EV'],
                'delta_max': result['delta_max'],
                'evaluation_time': end_time - start_time,
                'valid': not math.isinf(result['total'])
            }
        except Exception as e:
            end_time = time.time()
            return {
                'cost': float('inf'),
                'time': float('inf'),
                'energy': float('inf'),
                'ED': float('inf'),
                'EV': float('inf'),
                'delta_max': float('inf'),
                'evaluation_time': end_time - start_time,
                'valid': False,
                'error': str(e)
            }
    
    def compute_improvement(self, baseline_cost: float, method_cost: float) -> float:
        """Compute percentage improvement over baseline."""
        if math.isinf(baseline_cost) or baseline_cost == 0:
            return 0.0
        return ((baseline_cost - method_cost) / baseline_cost) * 100
    
    def aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results across multiple experiments."""
        by_method = defaultdict(lambda: defaultdict(list))
        
        for result in all_results:
            method = result['method']
            for metric in ['cost', 'time', 'energy', 'evaluation_time']:
                if metric in result and not math.isinf(result[metric]):
                    by_method[method][metric].append(result[metric])
        
        stats = {}
        for method, metrics in by_method.items():
            stats[method] = {}
            for metric, values in metrics.items():
                if values:
                    stats[method][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values)
                    }
        return stats
