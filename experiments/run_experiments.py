"""
Main Experiment Runner

Runs all experiments and collects results for research paper.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from experiments.baselines import BaselineMethods
from experiments.metrics import MetricsEvaluator


class ExperimentRunner:
    """Runs comprehensive experiments for research paper."""
    
    def __init__(self, agentic_system, utility_evaluator_class, workflow_class, environment_class, output_dir: str = "experiments/results"):
        self.system = agentic_system
        self.baselines = BaselineMethods(seed=42)
        self.metrics = MetricsEvaluator(utility_evaluator_class, workflow_class, environment_class)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_agentic(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run your agentic system on a problem."""
        start_time = time.time()
        try:
            result = self.system.execute(problem)
            end_time = time.time()
            return {
                'policy': result.get('optimal_policy', []),
                'cost': result.get('best_cost', float('inf')),
                'time_taken': end_time - start_time,
                'num_candidates': result.get('search_stats', {}).get('total_evaluated', 0),
                'num_iterations': result.get('search_stats', {}).get('iterations', 0),
                'valid': result.get('best_cost', float('inf')) != float('inf')
            }
        except Exception as e:
            end_time = time.time()
            print(f"Error in agentic system: {e}")
            return {'policy': [], 'cost': float('inf'), 'time_taken': end_time - start_time, 'num_candidates': 0, 'num_iterations': 0, 'valid': False, 'error': str(e)}
    
    def run_single_experiment(self, experiment: Dict[str, Any], run_baselines: bool = True, run_agentic: bool = True) -> List[Dict[str, Any]]:
        """Run single experiment comparing all methods."""
        workflow = experiment['workflow']
        env = experiment['env']
        params = experiment['params']
        metadata = experiment['metadata']
        num_tasks = metadata['num_tasks']
        num_locations = metadata['num_locations']
        results = []
        
        if run_baselines:
            baseline_policies = self.baselines.get_all_baselines(num_tasks, num_locations, workflow, env)
            for method_name, policy in baseline_policies.items():
                print(f"  Running {method_name}...", end=" ")
                metrics = self.metrics.evaluate_policy(policy, workflow, env, params)
                results.append({
                    'experiment_id': experiment['experiment_id'],
                    'method': method_name,
                    'num_tasks': num_tasks,
                    'num_locations': num_locations,
                    'policy': policy,
                    **metrics
                })
                print(f"Cost: {metrics['cost']:.2f}")
        
        if run_agentic:
            print(f"  Running agentic...", end=" ")
            agentic_result = self.run_agentic(experiment)
            if agentic_result['valid'] and agentic_result['policy']:
                detailed_metrics = self.metrics.evaluate_policy(agentic_result['policy'], workflow, env, params)
            else:
                detailed_metrics = {'cost': float('inf'), 'time': float('inf'), 'energy': float('inf'), 'valid': False}
            
            results.append({
                'experiment_id': experiment['experiment_id'],
                'method': 'agentic',
                'num_tasks': num_tasks,
                'num_locations': num_locations,
                'policy': agentic_result['policy'],
                'cost': agentic_result['cost'],
                'time': detailed_metrics.get('time', float('inf')),
                'energy': detailed_metrics.get('energy', float('inf')),
                'time_taken': agentic_result['time_taken'],
                'num_candidates': agentic_result['num_candidates'],
                'num_iterations': agentic_result['num_iterations'],
                'valid': agentic_result['valid']
            })
            print(f"Cost: {agentic_result['cost']:.2f}, Candidates: {agentic_result['num_candidates']}, Iterations: {agentic_result['num_iterations']}")
        
        return results
    
    def run_all_experiments(self, dataset: List[Dict[str, Any]], run_baselines: bool = True, run_agentic: bool = True) -> List[Dict[str, Any]]:
        """Run all experiments in dataset."""
        all_results = []
        print(f"Running {len(dataset)} experiments...\n")
        for i, experiment in enumerate(dataset, 1):
            exp_id = experiment['experiment_id']
            print(f"[{i}/{len(dataset)}] Experiment {exp_id}")
            results = self.run_single_experiment(experiment, run_baselines=run_baselines, run_agentic=run_agentic)
            all_results.extend(results)
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {filepath}")
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive statistics from results."""
        import numpy as np
        from collections import defaultdict
        
        stats = defaultdict(lambda: defaultdict(list))
        for result in results:
            method = result['method']
            num_tasks = result['num_tasks']
            if result.get('valid', True):
                stats[method][num_tasks].append({
                    'cost': result.get('cost', float('inf')),
                    'time': result.get('time', float('inf')),
                    'energy': result.get('energy', float('inf')),
                    'time_taken': result.get('time_taken', 0),
                    'num_candidates': result.get('num_candidates', 0)
                })
        
        aggregated = {}
        for method, by_size in stats.items():
            aggregated[method] = {}
            for size, entries in by_size.items():
                if not entries:
                    continue
                costs = [e['cost'] for e in entries if not np.isinf(e['cost'])]
                times = [e['time'] for e in entries if not np.isinf(e['time'])]
                energies = [e['energy'] for e in entries if not np.isinf(e['energy'])]
                time_takens = [e['time_taken'] for e in entries]
                candidates = [e['num_candidates'] for e in entries if e['num_candidates'] > 0]
                
                aggregated[method][size] = {
                    'cost_mean': np.mean(costs) if costs else float('inf'),
                    'cost_std': np.std(costs) if costs else 0,
                    'time_mean': np.mean(times) if times else float('inf'),
                    'energy_mean': np.mean(energies) if energies else float('inf'),
                    'time_taken_mean': np.mean(time_takens) if time_takens else 0,
                    'candidates_mean': np.mean(candidates) if candidates else 0,
                    'num_samples': len(entries)
                }
        return aggregated
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        for method, by_size in stats.items():
            print(f"\n{method.upper()}")
            print("-" * 80)
            print(f"{'Tasks':>6} {'Cost':>12} {'Time(ms)':>12} {'Energy(mJ)':>12} {'Runtime(s)':>12} {'Candidates':>12}")
            print("-" * 80)
            for size in sorted(by_size.keys()):
                s = by_size[size]
                print(f"{size:>6} {s['cost_mean']:>12.2f} {s['time_mean']:>12.2f} {s['energy_mean']:>12.2f} {s['time_taken_mean']:>12.3f} {s.get('candidates_mean', 0):>12.1f}")
