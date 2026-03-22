"""
Plotting and Visualization for Research Paper

Generates publication-quality plots for experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ExperimentPlotter:
    """Generate plots for research paper."""
    
    def __init__(self, output_dir: str = "experiments/results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11, 'figure.titlesize': 16})
    
    def load_results(self, filename: str):
        with open(filename, 'r') as f:
            return json.load(f)
    
    def plot_cost_vs_tasks(self, results, filename: str = "cost_vs_tasks.pdf"):
        import pandas as pd
        df = pd.DataFrame(results)
        df = df[df['valid'] == True]
        grouped = df.groupby(['num_tasks', 'method'])['cost'].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.plot(ax=ax, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Total Cost U(w,p)')
        ax.set_title('Cost vs DAG Size: Comparison of Methods')
        ax.legend(title='Method', loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_runtime_vs_tasks(self, results, filename: str = "runtime_vs_tasks.pdf"):
        import pandas as pd
        df = pd.DataFrame(results)
        df = df[df['valid'] == True]
        df_with_time = df[df['time_taken'] > 0]
        grouped = df_with_time.groupby(['num_tasks', 'method'])['time_taken'].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.plot(ax=ax, marker='s', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Computational Time vs DAG Size')
        ax.set_yscale('log')
        ax.legend(title='Method', loc='best')
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_candidates_evaluated(self, results, filename: str = "candidates_evaluated.pdf"):
        import pandas as pd
        df = pd.DataFrame(results)
        df = df[(df['method'] == 'agentic') & (df['num_candidates'] > 0)]
        if df.empty:
            print("No candidate data available")
            return
        
        grouped = df.groupby('num_tasks')['num_candidates'].agg(['mean', 'std'])
        num_locations = 3
        exhaustive = [num_locations ** n for n in grouped.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], label='Agentic (Iterative)', marker='o', linewidth=2, capsize=5)
        ax.plot(grouped.index, exhaustive, label='Exhaustive Search', marker='x', linewidth=2, linestyle='--')
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Candidates Evaluated')
        ax.set_title('Search Efficiency: Agentic vs Exhaustive')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_improvement_over_baseline(self, results, baseline: str = 'random', filename: str = "improvement.pdf"):
        import pandas as pd
        df = pd.DataFrame(results)
        df = df[df['valid'] == True]
        improvements = []
        
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            baseline_cost = exp_data[exp_data['method'] == baseline]['cost'].values
            agentic_cost = exp_data[exp_data['method'] == 'agentic']['cost'].values
            if len(baseline_cost) > 0 and len(agentic_cost) > 0:
                baseline_cost = baseline_cost[0]
                agentic_cost = agentic_cost[0]
                if baseline_cost > 0 and not np.isinf(baseline_cost):
                    improvement = ((baseline_cost - agentic_cost) / baseline_cost) * 100
                    improvements.append({'num_tasks': exp_data['num_tasks'].values[0], 'improvement': improvement})
        
        if not improvements:
            print("No improvement data available")
            return
        
        imp_df = pd.DataFrame(improvements)
        grouped = imp_df.groupby('num_tasks')['improvement'].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(grouped.index, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7, color='steelblue')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel(f'Improvement over {baseline.capitalize()} (%)')
        ax.set_title(f'Cost Reduction: Agentic vs {baseline.capitalize()} Baseline')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_convergence(self, results, filename: str = "convergence.pdf"):
        import pandas as pd
        df = pd.DataFrame(results)
        df = df[(df['method'] == 'agentic') & (df['num_iterations'] > 0)]
        if df.empty:
            print("No convergence data available")
            return
        
        grouped = df.groupby('num_tasks')['num_iterations'].agg(['mean', 'std'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(grouped.index, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7, color='coral')
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Iterations to Convergence')
        ax.set_title('Search Convergence Speed')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def generate_all_plots(self, results_file: str):
        print("Generating plots for research paper...\n")
        results = self.load_results(results_file)
        self.plot_cost_vs_tasks(results)
        self.plot_runtime_vs_tasks(results)
        self.plot_candidates_evaluated(results)
        self.plot_improvement_over_baseline(results, baseline='random')
        self.plot_improvement_over_baseline(results, baseline='all_cloud', filename='improvement_cloud.pdf')
        self.plot_convergence(results)
        print(f"\nAll plots saved to {self.output_dir}/")
