#!/usr/bin/env python3
"""
Quick Start Guide - Agentic Task Offloading System

This script demonstrates a minimal working example of the system.
Run this first before running full experiments.
"""

import os
import sys
from pathlib import Path

# Add repo to path
sys.path.append(str(Path(__file__).parent))

from agents.config import create_system


def create_minimal_example():
    """Create a small example for quick testing."""
    return {
        "env": {
            "locations": {0: "iot", 1: "edge", 2: "cloud"},
            "DR": {
                (0, 0): 0.0,
                (0, 1): 1e-6,
                (0, 2): 3e-6,
                (1, 0): 1e-6,
                (1, 1): 0.0,
                (1, 2): 2.5e-6,
                (2, 0): 3e-6,
                (2, 1): 2.5e-6,
                (2, 2): 0.0,
            },
            "DE": {0: 1e-6, 1: 5e-7, 2: 1e-7},
            "VR": {0: 1e-6, 1: 2e-7, 2: 1e-7},
            "VE": {0: 1e-6, 1: 3e-7, 2: 5e-8},
        },
        "workflow": {
            "N": 3,
            "tasks": {
                1: {"v": 1e7},
                2: {"v": 5e6},
                3: {"v": 8e6},
            },
            "edges": {
                (0, 1): 0.0,
                (1, 2): 2e6,
                (2, 3): 1.5e6,
                (3, 4): 0.0,
            },
        },
        "params": {
            "CT": 0.2,
            "CE": 1.34,
            "delta_t": 1,
            "delta_e": 1,
        },
    }


def main():
    print("=" * 70)
    print("QUICK START - AGENTIC TASK OFFLOADING")
    print("=" * 70)

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Missing GOOGLE_API_KEY environment variable")
        print("\nSetup:")
        print("  1. Get API key from: https://aistudio.google.com/app/apikeys")
        print("  2. Set environment variable:")
        print("     - Windows: $env:GOOGLE_API_KEY = 'sk-...'")
        print("     - Linux/Mac: export GOOGLE_API_KEY='sk-...'")
        return

    print("\n✓ API key found")

    # Create system
    print("\n[1/3] Creating agentic system...")
    try:
        orchestrator = create_system(api_key=api_key)
        print("✓ System initialized (Planner + Evaluator + Output)")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Load example problem
    print("\n[2/3] Loading example problem (3 tasks, 3 locations)...")
    problem = create_minimal_example()
    print("✓ Problem loaded")

    # Run
    print("\n[3/3] Running optimization...")
    try:
        result = orchestrator.execute(problem)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if result.get("optimal_policy"):
            print("\n✓ Optimal Policy Found:")
            print(f"  Placements: {result['optimal_policy']}")
            print(f"  Best Cost:  {result.get('best_cost', 'N/A'):.6f}")
            print("\nTask Assignments:")
            for i, loc in enumerate(result["optimal_policy"], 1):
                loc_name = problem["env"]["locations"].get(loc, "unknown")
                print(f"  Task {i} → Location {loc} ({loc_name})")
        else:
            print("\n❌ No solution found:")
            print(f"  {result.get('evaluation', 'Unknown error')}")

        print("\nAgent Logs: agent_trace.txt")

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("✓ Quick start complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review agent_trace.txt for reasoning")
    print("  2. Try larger workflows (5-10 tasks)")
    print("  3. Run full experiments: python experiments/run_all.py")


if __name__ == "__main__":
    main()
