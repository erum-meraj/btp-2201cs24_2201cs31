"""
Master Script: Run All Experiments for Research Paper

Usage: python experiments/run_all.py <api_key>

Requires: pandas, matplotlib, seaborn
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from experiments.dataset import DatasetGenerator
from experiments.run_experiments import ExperimentRunner
from experiments.plot_results import ExperimentPlotter

# Import core modules and agent system
from core.workflow import Workflow
from core.environment import Environment
from core.cost_eval import UtilityEvaluator
from core.memory_manager import WorkflowMemory
from agents.config import create_system


def create_agentic_system(api_key: str):
    """
    Create the complete agentic system for task offloading optimization.

    Args:
        api_key: Google Gemini API key for LLM-based agents

    Returns:
        AgentOrchestrator instance ready to execute workflows
    """
    # Ensure memory directory exists
    memory_dir = "experiments/memory_store"
    os.makedirs(memory_dir, exist_ok=True)

    # Initialize memory manager for few-shot learning
    memory_manager = WorkflowMemory(memory_dir=memory_dir)

    # Create orchestrator using standard system builder
    # This initializes: planner, evaluator, output agents
    orchestrator = create_system(
        api_key=api_key,
        log_file="experiments/experiment_trace.log",
        memory_manager=memory_manager,
    )

    return orchestrator


def main():
    print("=" * 80)
    print("RESEARCH PAPER EXPERIMENTS - AGENTIC TASK OFFLOADING")
    print("=" * 80)

    # Get API key from environment or command line
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            print("✗ No API key provided")
            print("\nUsage: python run_all.py <api_key>")
            print("Or set GOOGLE_API_KEY environment variable")
            sys.exit(1)

    # Create output directory
    OUTPUT_DIR = "experiments/results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    # Step 1: Generate dataset
    print("\n[1/5] Generating benchmark dataset...")
    generator = DatasetGenerator(seed=42)
    dataset = generator.create_dataset(
        task_sizes=[5, 7, 10, 15, 20], samples_per_size=10, num_locations=3
    )
    generator.save_dataset(dataset, f"{OUTPUT_DIR}/benchmark_dataset.json")
    print(f"✓ Generated {len(dataset)} experiments")

    # Step 2: Create system
    print("\n[2/5] Initializing agentic system...")
    try:
        orchestrator = create_agentic_system(api_key)
        print("✓ Orchestrator initialized with Planner, Evaluator, Output agents")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Run experiments
    print("\n[3/5] Running experiments...")
    print("This may take 10-30 minutes depending on dataset size...\n")

    try:
        runner = ExperimentRunner(
            orchestrator, UtilityEvaluator, Workflow, Environment, OUTPUT_DIR
        )
        all_results = runner.run_all_experiments(
            dataset, run_baselines=True, run_agentic=True
        )
        runner.save_results(all_results, f"{OUTPUT_DIR}/all_results.json")
        print("✓ Experiments completed and results saved")

    except Exception as e:
        print(f"✗ Error during experiment execution: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Compute statistics
    print("\n[4/5] Computing statistics...")
    try:
        stats = runner.compute_statistics(all_results)
        runner.print_summary(stats)

        import json

        with open(f"{OUTPUT_DIR}/statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("✓ Statistics computed and saved")

    except Exception as e:
        print(f"✗ Error computing statistics: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 5: Generate plots
    print("\n[5/5] Generating plots...")
    try:
        plotter = ExperimentPlotter(output_dir=f"{OUTPUT_DIR}/plots")
        plotter.generate_all_plots(f"{OUTPUT_DIR}/all_results.json")
        print("✓ All plots generated successfully")

    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"\nResults in: {OUTPUT_DIR}/")
    print("  - all_results.json")
    print("  - statistics.json")
    print("  - plots/")


if __name__ == "__main__":
    # Check for required packages at startup
    try:
        import importlib.util

        required = ["pandas", "matplotlib", "seaborn"]
        missing = []
        for pkg in required:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)

        if missing:
            print("Missing required packages:", ", ".join(missing))
            print(f"\nInstall via: pip install {' '.join(missing)}")
            sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not verify packages: {e}")

    main()
