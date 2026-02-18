"""
Example Usage Script

Demonstrates how to use the refactored agentic task offloading system.
"""

import os
import sys
import json
import dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dotenv.load_dotenv()  # Load environment variables from .env file

from config import create_system


def create_example_problem():
    """
    Create an example task offloading problem.

    Returns:
        Dictionary with env, workflow, and params
    """
    # Example environment: IoT device (0), 2 edge servers (1,2), 1 cloud (3)
    env = {
        "locations": {0: "iot", 1: "edge", 2: "edge", 3: "cloud"},
        "DR": {  # Data transfer time (ms/byte)
            (0, 0): 0.0,
            (0, 1): 1e-6,  # IoT to edge 1
            (0, 2): 1.2e-6,  # IoT to edge 2
            (0, 3): 3e-6,  # IoT to cloud
            (1, 0): 1e-6,
            (1, 1): 0.0,
            (1, 2): 8e-7,
            (1, 3): 2.5e-6,
            (2, 0): 1.2e-6,
            (2, 1): 8e-7,
            (2, 2): 0.0,
            (2, 3): 2.8e-6,
            (3, 0): 3e-6,
            (3, 1): 2.5e-6,
            (3, 2): 2.8e-6,
            (3, 3): 0.0,
        },
        "DE": {0: 1e-6, 1: 5e-7, 2: 5e-7, 3: 1e-7},  # Data energy (mJ/byte)
        "VR": {  # Task time (ms/cycle)
            0: 1e-6,  # Slow IoT
            1: 2e-7,  # Fast edge
            2: 2.5e-7,
            3: 1e-7,  # Very fast cloud
        },
        "VE": {0: 1e-6, 1: 3e-7, 2: 3e-7, 3: 5e-8},  # Task energy (mJ/cycle)
    }

    # Example workflow: 4-task DAG
    # Task 1 -> Task 2 -> Task 4
    #       \-> Task 3 -/
    workflow = {
        "N": 4,
        "tasks": {
            1: {"v": 1e7},  # 10M cycles
            2: {"v": 5e6},  # 5M cycles
            3: {"v": 8e6},  # 8M cycles
            4: {"v": 1.5e7},  # 15M cycles
        },
        "edges": {
            (0, 1): 0.0,  # Entry to task 1
            (1, 2): 2e6,  # Task 1 to 2 (2MB)
            (1, 3): 1.5e6,  # Task 1 to 3 (1.5MB)
            (2, 4): 3e6,  # Task 2 to 4 (3MB)
            (3, 4): 2.5e6,  # Task 3 to 4 (2.5MB)
            (4, 5): 0.0,  # Task 4 to exit
        },
    }

    # Cost parameters - Balanced mode
    params = {
        "CT": 0.2,  # Cost per ms
        "CE": 1.34,  # Cost per mJ
        "delta_t": 1,  # Time weight
        "delta_e": 1,  # Energy weight
    }

    return {"env": env, "workflow": workflow, "params": params}


def run_example(api_key=None):
    """
    Run a complete example of the agentic system.

    Args:
        api_key: Optional API key (otherwise reads from environment)
    """
    print("=" * 70)
    print("AGENTIC TASK OFFLOADING SYSTEM - EXAMPLE EXECUTION")
    print("=" * 70)

    # Create the system
    print("\n[1/3] Initializing agent system...")
    try:
        orchestrator = create_system(
            api_key=api_key, log_file="trace_files/example_trace.txt"
        )
        print("  ✓ System initialized successfully")
    except Exception as e:
        print(f"  ✗ Error initializing system: {e}")
        print("\nPlease ensure:")
        print("  1. GOOGLE_API_KEY environment variable is set")
        print("  2. All required modules are in the path")
        return

    # Create example problem
    print("\n[2/3] Creating example problem...")
    problem = create_example_problem()
    print(f"  ✓ Problem created:")
    print(f"     - {problem['workflow']['N']} tasks")
    print(f"     - {len(problem['env']['locations'])} locations")
    print(f"     - Mode: Balanced (optimize time and energy)")

    # Execute the system
    print("\n[3/3] Executing agent workflow...")
    print("-" * 70)

    try:
        result = orchestrator.execute(problem)

        # Display results
        print("\n" + "=" * 70)
        print("EXECUTION COMPLETE - RESULTS")
        print("=" * 70)

        if result.get("optimal_policy"):
            print(f"\nOptimal Policy: {result['optimal_policy']}")
            print(f"Total Cost: {result.get('best_cost', 'N/A')}")
            print(f"\nTask Placements:")
            for i, loc in enumerate(result["optimal_policy"], 1):
                loc_type = problem["env"]["locations"].get(loc, "unknown")
                print(f"  Task {i} -> Location {loc} ({loc_type})")
        else:
            print("\nNo optimal policy found")
            print(f"Reason: {result.get('evaluation', 'Unknown')}")

        print("\n" + "=" * 70)
        print(f"Full trace saved to: example_trace.txt")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_multiple_modes(api_key=None):
    """
    Run the same problem in different optimization modes.

    Args:
        api_key: Optional API key
    """
    print("=" * 70)
    print("TESTING DIFFERENT OPTIMIZATION MODES")
    print("=" * 70)

    # Create system once
    orchestrator = create_system(api_key=api_key, log_file="modes_trace.txt")
    base_problem = create_example_problem()

    modes = [
        ("Balanced", {"delta_t": 1, "delta_e": 1}),
        ("Low Latency", {"delta_t": 1, "delta_e": 0}),
        ("Low Power", {"delta_t": 0, "delta_e": 1}),
    ]

    results = {}

    for mode_name, mode_params in modes:
        print(f"\n{'='*70}")
        print(f"Testing {mode_name} Mode")
        print(f"{'='*70}")

        # Update parameters for this mode
        problem = base_problem.copy()
        problem["params"].update(mode_params)

        try:
            result = orchestrator.execute(problem)
            results[mode_name] = result
            print(f"\n✓ {mode_name} mode complete")
            if result.get("optimal_policy"):
                print(f"  Policy: {result['optimal_policy']}")
                print(f"  Cost: {result.get('best_cost', 'N/A'):.6f}")
        except Exception as e:
            print(f"\n✗ Error in {mode_name} mode: {e}")
            results[mode_name] = None

    # Compare results
    print("\n" + "=" * 70)
    print("MODE COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<15} {'Policy':<25} {'Cost':<15}")
    print("-" * 70)
    for mode_name, result in results.items():
        if result and result.get("optimal_policy"):
            policy_str = str(result["optimal_policy"])
            cost_str = f"{result.get('best_cost', 0):.6f}"
        else:
            policy_str = "N/A"
            cost_str = "N/A"
        print(f"{mode_name:<15} {policy_str:<25} {cost_str:<15}")

    return results


if __name__ == "__main__":
    # Check if API key is provided as command line argument
    api_key = sys.argv[1] if len(sys.argv) > 1 else None

    # Run single example
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single execution with balanced mode")
    print("=" * 70)
    run_example(api_key)

    # Optionally run multiple modes
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Compare different optimization modes")
    print("=" * 70)
    response = input("\nRun multi-mode comparison? (y/n): ")
    if response.lower() == "y":
        run_multiple_modes(api_key)
