"""
Architecture Verification Test

Tests the basic structure and interfaces of the agentic system.
"""

import sys
import os

# Ensure root is in path to import 'agents' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from agents.orchestrator.orchestrator import AgentOrchestrator

        print("  ✓ orchestrator.py")
    except ImportError as e:
        print(f"  ✗ orchestrator.py: {e}")
        return False

    try:
        from agents.evaluator_agent.evaluator import EvaluatorAgent

        print("  ✓ evaluator_agent.py")
    except ImportError as e:
        print(f"  ✗ evaluator_agent.py: {e}")
        return False

    try:
        from agents.evaluator_agent.candidate_generator.candidate_generator import (
            CandidatePolicyGenerator,
        )

        print("  ✓ candidate_generator.py")
    except ImportError as e:
        print(f"  ✗ candidate_generator.py: {e}")
        return False

    try:
        from agents.evaluator_agent.tools.utility_function import UtilityFunctionTool

        print("  ✓ utility_tool.py")
    except ImportError as e:
        print(f"  ✗ utility_tool.py: {e}")
        return False

    try:
        from agents.evaluator_agent.weak_solver.weak_solver import WeakSolverTool

        print("  ✓ weak_solver.py")
    except ImportError as e:
        print(f"  ✗ weak_solver.py: {e}")
        return False

    try:
        from agents.config import AgentConfig, SystemBuilder, create_system

        print("  ✓ config.py")
    except ImportError as e:
        print(f"  ✗ config.py: {e}")
        return False

    print("✓ All modules imported successfully\n")
    return True


def test_tool_interfaces():
    """Test that tools have the expected interfaces."""
    print("Testing tool interfaces...")

    from agents.evaluator_agent.candidate_generator import CandidatePolicyGenerator
    from agents.evaluator_agent.tools.utility_function import UtilityFunctionTool
    from agents.evaluator_agent.weak_solver.weak_solver import WeakSolverTool

    # Test CandidatePolicyGenerator
    generator = CandidatePolicyGenerator()
    assert hasattr(
        generator, "generate_candidates"
    ), "Missing generate_candidates method"
    assert hasattr(
        generator, "filter_by_constraints"
    ), "Missing filter_by_constraints method"
    print("  ✓ CandidatePolicyGenerator interface")

    # Test WeakSolverTool
    solver = WeakSolverTool()
    assert hasattr(solver, "solve"), "Missing solve method"
    assert hasattr(solver, "enable"), "Missing enable method"
    assert hasattr(solver, "is_enabled"), "Missing is_enabled method"
    print("  ✓ WeakSolverTool interface")

    print("✓ All tool interfaces valid\n")
    return True


def test_agent_interfaces():
    """Test that agents have the expected run method."""
    print("Testing agent interfaces...")

    # Note: We can't fully instantiate agents without API keys,
    # but we can check the class definitions

    from agents.evaluator_agent.evaluator import EvaluatorAgent

    assert hasattr(EvaluatorAgent, "run"), "EvaluatorAgent missing run method"
    assert hasattr(
        EvaluatorAgent, "find_best_policy"
    ), "EvaluatorAgent missing find_best_policy"
    print("  ✓ EvaluatorAgent interface")

    from agents.orchestrator.orchestrator import AgentOrchestrator

    assert hasattr(AgentOrchestrator, "execute"), "Orchestrator missing execute method"
    print("  ✓ AgentOrchestrator interface")

    print("✓ All agent interfaces valid\n")
    return True


def test_configuration():
    """Test configuration module."""
    print("Testing configuration...")

    from agents.config import AgentConfig, SystemBuilder

    # Test AgentConfig creation (without actual API key)
    try:
        config = AgentConfig(
            api_key="test-key",
            log_file="test.txt",
            model_name="test-model",
            temperature=0.5,
        )
        assert config.api_key == "test-key"
        assert config.log_file == "test.txt"
        assert config.temperature == 0.5
        print("  ✓ AgentConfig creation")
    except Exception as e:
        print(f"  ✗ AgentConfig creation: {e}")
        return False

    print("✓ Configuration module valid\n")
    return True


def test_candidate_generation():
    """Test candidate generation logic."""
    print("Testing candidate generation...")

    from agents.evaluator_agent.candidate_generator.candidate_generator import (
        CandidatePolicyGenerator,
    )

    generator = CandidatePolicyGenerator()

    # Test systematic candidate generation
    num_tasks = 3
    location_ids = [0, 1, 2]

    systematic = generator._generate_systematic_candidates(num_tasks, location_ids)

    # Should have at least one candidate per location (all tasks to that location)
    assert len(systematic) >= len(location_ids), "Not enough systematic candidates"

    # Check that candidates have correct length
    for policy in systematic:
        assert len(policy) == num_tasks, f"Policy {policy} has wrong length"

    print(f"  ✓ Generated {len(systematic)} systematic candidates")

    # Test deduplication
    duplicates = [(0, 0, 0), (1, 1, 1), (0, 0, 0), (2, 2, 2)]
    unique = generator._deduplicate(duplicates)
    assert len(unique) == 3, "Deduplication failed"
    print("  ✓ Deduplication works")

    # Test constraint checking
    policy = (0, 1, 2)
    fixed = {1: 0}  # Task 1 must be at location 0
    assert generator._satisfies_constraints(
        policy, fixed, None
    ), "Should satisfy fixed constraint"

    bad_policy = (1, 1, 2)
    assert not generator._satisfies_constraints(
        bad_policy, fixed, None
    ), "Should violate fixed constraint"
    print("  ✓ Constraint checking works")

    print("✓ Candidate generation valid\n")
    return True


def test_weak_solver():
    """Test weak solver placeholder."""
    print("Testing weak solver...")

    from agents.evaluator_agent.weak_solver.weak_solver import WeakSolverTool

    solver = WeakSolverTool()

    # Should start disabled
    assert not solver.is_enabled(), "Solver should start disabled"
    print("  ✓ Starts disabled")

    # Test enable/disable
    solver.enable(["genetic_algorithm"])
    assert solver.is_enabled(), "Solver should be enabled"
    assert "genetic_algorithm" in solver.algorithms
    print("  ✓ Enable/disable works")

    solver.disable()
    assert not solver.is_enabled(), "Solver should be disabled"
    print("  ✓ State management works")

    print("✓ Weak solver valid\n")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("AGENTIC SYSTEM ARCHITECTURE VERIFICATION")
    print("=" * 70)
    print()

    tests = [
        ("Module Imports", test_imports),
        ("Tool Interfaces", test_tool_interfaces),
        ("Agent Interfaces", test_agent_interfaces),
        ("Configuration", test_configuration),
        ("Candidate Generation", test_candidate_generation),
        ("Weak Solver", test_weak_solver),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Architecture is valid.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
