"""
Dataset Generation for Experiments

Generates synthetic workflows and environments for benchmarking.
"""

import random
import numpy as np
from typing import Dict, List, Any
import json


class DatasetGenerator:
    """Generate synthetic task offloading problems for experiments."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_dag(self, num_tasks: int, edge_prob: float = 0.3) -> Dict[int, List[int]]:
        """Generate a random Directed Acyclic Graph (DAG)."""
        dag = {i: [] for i in range(num_tasks)}
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                if random.random() < edge_prob:
                    dag[i].append(j)
        return dag
    
    def generate_workflow(self, num_tasks: int) -> Dict[str, Any]:
        """Generate a random workflow (application DAG)."""
        dag = self.generate_dag(num_tasks, edge_prob=0.3)
        task_sizes = np.random.uniform(1e6, 1e8, size=num_tasks)
        tasks = {i + 1: {'v': task_sizes[i]} for i in range(num_tasks)}
        
        edges = {(0, 1): 0.0}
        for i in range(num_tasks):
            if dag[i]:
                for j in dag[i]:
                    data_size = np.random.uniform(1e5, 1e7)
                    edges[(i + 1, j + 1)] = data_size
            else:
                edges[(i + 1, num_tasks + 2)] = 0.0
        edges[(num_tasks + 1, num_tasks + 2)] = 0.0
        
        return {'N': num_tasks, 'tasks': tasks, 'edges': edges}
    
    def generate_environment(self, num_locations: int = 3) -> Dict[str, Any]:
        """Generate random edge-cloud environment parameters."""
        locations = {0: 'iot', 1: 'edge', 2: 'cloud'}
        base_rates = {'iot': 5e-6, 'edge': 2e-6, 'cloud': 1e-6}
        
        DR = {}
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    DR[(i, j)] = 0.0
                else:
                    avg_rate = (base_rates[locations[i]] + base_rates[locations[j]]) / 2
                    DR[(i, j)] = avg_rate * np.random.uniform(0.8, 1.2)
        
        DE = {0: np.random.uniform(8e-7, 1.2e-6), 1: np.random.uniform(3e-7, 6e-7), 2: np.random.uniform(5e-8, 2e-7)}
        VR = {0: np.random.uniform(8e-7, 1.2e-6), 1: np.random.uniform(1e-7, 3e-7), 2: np.random.uniform(5e-8, 1.5e-7)}
        VE = {0: np.random.uniform(8e-7, 1.2e-6), 1: np.random.uniform(2e-7, 5e-7), 2: np.random.uniform(3e-8, 1.5e-7)}
        
        return {'locations': locations, 'DR': DR, 'DE': DE, 'VR': VR, 'VE': VE}
    
    def create_dataset(self, task_sizes: List[int] = [5, 7, 10, 15], samples_per_size: int = 10, num_locations: int = 3) -> List[Dict[str, Any]]:
        """Create a complete benchmark dataset."""
        dataset = []
        for num_tasks in task_sizes:
            for sample_id in range(samples_per_size):
                dataset.append({
                    'experiment_id': f"n{num_tasks}_s{sample_id}",
                    'workflow': self.generate_workflow(num_tasks),
                    'env': self.generate_environment(num_locations),
                    'params': {'CT': 0.2, 'CE': 1.34, 'delta_t': 1, 'delta_e': 1},
                    'metadata': {'num_tasks': num_tasks, 'num_locations': num_locations, 'sample_id': sample_id}
                })
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to JSON file."""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} experiments to {filename}")


if __name__ == "__main__":
    generator = DatasetGenerator(seed=42)
    dataset = generator.create_dataset(task_sizes=[5, 7, 10, 15, 20], samples_per_size=10)
    generator.save_dataset(dataset, "benchmark_dataset.json")
