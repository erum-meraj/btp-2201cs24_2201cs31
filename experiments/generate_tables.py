"""
Generate LaTeX Tables for Research Paper

Creates publication-ready LaTeX tables from experiment results.
"""

import json
import numpy as np
from pathlib import Path


class LaTeXTableGenerator:
    """Generate LaTeX tables from experiment results."""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
    
    def load_statistics(self):
        with open(self.results_dir / "statistics.json", 'r') as f:
            return json.load(f)
    
    def generate_cost_comparison_table(self, stats, filename: str = "cost_comparison.tex"):
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Cost Comparison: Average Total Cost $U(w,p)$ (Mean $\\pm$ Std)}")
        latex.append("\\label{tab:cost_comparison}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Method & 5 Tasks & 10 Tasks & 15 Tasks & 20 Tasks \\\\")
        latex.append("\\midrule")
        
        methods = sorted(stats.keys())
        task_sizes = [5, 10, 15, 20]
        
        for method in methods:
            method_name = method.replace('_', ' ').title()
            row = [method_name]
            for size in task_sizes:
                if size in stats[method]:
                    cost_mean = stats[method][size]['cost_mean']
                    cost_std = stats[method][size]['cost_std']
                    row.append(f"${cost_mean:.1f} \\pm {cost_std:.1f}$")
                else:
                    row.append("---")
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        output = "\n".join(latex)
        with open(self.results_dir / filename, 'w') as f:
            f.write(output)
        print(f"Generated: {filename}")
        return output
    
    def generate_efficiency_table(self, stats, filename: str = "efficiency.tex"):
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Search Efficiency: Candidates Evaluated}")
        latex.append("\\label{tab:efficiency}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Tasks & Exhaustive & Agentic & Reduction (\\%) \\\\")
        latex.append("\\midrule")
        
        task_sizes = [5, 7, 10, 15, 20]
        num_locations = 3
        
        for size in task_sizes:
            exhaustive = num_locations ** size
            if 'agentic' in stats and size in stats['agentic']:
                agentic_mean = stats['agentic'][size].get('candidates_mean', 0)
                if agentic_mean > 0:
                    reduction = ((exhaustive - agentic_mean) / exhaustive) * 100
                    if exhaustive > 1e6:
                        exhaustive_str = f"{exhaustive/1e6:.1f}M"
                    elif exhaustive > 1e3:
                        exhaustive_str = f"{exhaustive/1e3:.1f}K"
                    else:
                        exhaustive_str = f"{exhaustive}"
                    latex.append(f"{size} & {exhaustive_str} & {agentic_mean:.0f} & {reduction:.2f}\\% \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        output = "\n".join(latex)
        with open(self.results_dir / filename, 'w') as f:
            f.write(output)
        print(f"Generated: {filename}")
        return output
    
    def generate_improvement_table(self, stats, baseline: str = 'random', filename: str = "improvement.tex"):
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Cost Improvement over {baseline.title()} Baseline (\\%)}}")
        latex.append("\\label{tab:improvement}")
        latex.append("\\begin{tabular}{lccccc}")
        latex.append("\\toprule")
        latex.append("Tasks & Cost & Time & Energy & Runtime \\\\")
        latex.append("\\midrule")
        
        task_sizes = [5, 10, 15, 20]
        
        for size in task_sizes:
            if baseline in stats and 'agentic' in stats:
                if size in stats[baseline] and size in stats['agentic']:
                    baseline_cost = stats[baseline][size]['cost_mean']
                    agentic_cost = stats['agentic'][size]['cost_mean']
                    baseline_time = stats[baseline][size].get('time_mean', 0)
                    agentic_time = stats['agentic'][size].get('time_mean', 0)
                    baseline_energy = stats[baseline][size].get('energy_mean', 0)
                    agentic_energy = stats['agentic'][size].get('energy_mean', 0)
                    baseline_runtime = stats[baseline][size].get('time_taken_mean', 0)
                    agentic_runtime = stats['agentic'][size].get('time_taken_mean', 0)
                    
                    cost_imp = ((baseline_cost - agentic_cost) / baseline_cost) * 100
                    time_imp = ((baseline_time - agentic_time) / baseline_time) * 100 if baseline_time > 0 else 0
                    energy_imp = ((baseline_energy - agentic_energy) / baseline_energy) * 100 if baseline_energy > 0 else 0
                    runtime_change = ((agentic_runtime - baseline_runtime) / baseline_runtime) * 100 if baseline_runtime > 0 else 0
                    
                    latex.append(f"{size} & {cost_imp:.1f}\\% & {time_imp:.1f}\\% & {energy_imp:.1f}\\% & {runtime_change:+.1f}\\% \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        output = "\n".join(latex)
        with open(self.results_dir / filename, 'w') as f:
            f.write(output)
        print(f"Generated: {filename}")
        return output
    
    def generate_all_tables(self):
        print("Generating LaTeX tables...\n")
        stats = self.load_statistics()
        self.generate_cost_comparison_table(stats)
        self.generate_efficiency_table(stats)
        self.generate_improvement_table(stats, baseline='random')
        self.generate_improvement_table(stats, baseline='all_cloud', filename='improvement_cloud.tex')
        print(f"\nTables saved to {self.results_dir}/")


if __name__ == "__main__":
    generator = LaTeXTableGenerator()
    generator.generate_all_tables()
