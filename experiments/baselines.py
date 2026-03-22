"""
Baseline Methods for Comparison

Implements standard heuristic approaches for task offloading.
"""

import random
from typing import List, Dict, Any


class BaselineMethods:
    """Collection of baseline heuristic methods."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def random_policy(self, num_tasks: int, num_locations: int) -> List[int]:
        """Random placement policy."""
        return [random.randint(0, num_locations - 1) for _ in range(num_tasks)]
    
    def all_local(self, num_tasks: int) -> List[int]:
        """All-local policy (all tasks on IoT device)."""
        return [0] * num_tasks
    
    def all_edge(self, num_tasks: int) -> List[int]:
        """All-edge policy (all tasks on edge server)."""
        return [1] * num_tasks
    
    def all_cloud(self, num_tasks: int, cloud_id: int = 2) -> List[int]:
        """All-cloud policy (all tasks on cloud)."""
        return [cloud_id] * num_tasks
    
    def round_robin(self, num_tasks: int, num_locations: int) -> List[int]:
        """Round-robin policy (distribute cyclically)."""
        return [i % num_locations for i in range(num_tasks)]
    
    def greedy_computation(self, workflow: Dict[str, Any], env: Dict[str, Any]) -> List[int]:
        """Greedy policy based on computation intensity."""
        num_tasks = workflow['N']
        tasks = workflow['tasks']
        task_loads = [tasks[i+1]['v'] for i in range(num_tasks)]
        median_load = sorted(task_loads)[num_tasks // 2]
        
        policy = []
        for i in range(num_tasks):
            load = task_loads[i]
            if load > median_load * 1.5:
                policy.append(2)  # Heavy → Cloud
            elif load < median_load * 0.5:
                policy.append(0)  # Light → Local
            else:
                policy.append(1)  # Medium → Edge
        return policy
    
    def greedy_data(self, workflow: Dict[str, Any], env: Dict[str, Any]) -> List[int]:
        """Greedy policy based on data dependencies."""
        num_tasks = workflow['N']
        edges = workflow['edges']
        
        data_transfer = {}
        for (u, v), size in edges.items():
            if 1 <= u <= num_tasks and 1 <= v <= num_tasks:
                data_transfer[u] = data_transfer.get(u, 0) + size
        
        policy = []
        for i in range(num_tasks):
            task_id = i + 1
            transfer = data_transfer.get(task_id, 0)
            if transfer > 5e6:
                policy.append(1)  # High data → Edge
            elif transfer < 1e6:
                policy.append(2)  # Low data → Cloud
            else:
                policy.append(1)  # Medium → Edge
        return policy
    
    def get_all_baselines(self, num_tasks: int, num_locations: int, workflow: Dict[str, Any] = None, env: Dict[str, Any] = None) -> Dict[str, List[int]]:
        """Get all baseline policies for comparison."""
        baselines = {
            'random': self.random_policy(num_tasks, num_locations),
            'all_local': self.all_local(num_tasks),
            'all_cloud': self.all_cloud(num_tasks),
            'round_robin': self.round_robin(num_tasks, num_locations)
        }
        
        if num_locations >= 2:
            baselines['all_edge'] = self.all_edge(num_tasks)
        
        if workflow and env:
            baselines['greedy_compute'] = self.greedy_computation(workflow, env)
            baselines['greedy_data'] = self.greedy_data(workflow, env)
        
        return baselines
