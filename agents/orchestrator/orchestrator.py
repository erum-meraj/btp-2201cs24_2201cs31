"""
Orchestrator for the multi-agent task offloading system.

This module coordinates the flow between:
1. Planner Agent (strategic planning)
2. Evaluator Agent (policy search with candidate generation and weak solver)
3. Output Agent (result formatting)
"""

from typing import Dict, Any, Optional
import logging


class AgentOrchestrator:
    """
    Coordinates the execution flow of the multi-agent system.

    Flow:
    1. Planner generates strategic plan
    2. Evaluator searches for optimal policy using:
       - Candidate policy generator
       - Memory-based similar cases
       - Utility function evaluation
       - Weak solver (placeholder)
    3. Output agent formats and presents results
    """

    def __init__(
        self,
        planner_agent,
        evaluator_agent,
        output_agent,
        memory_manager=None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the orchestrator with agent instances.

        Args:
            planner_agent: Strategic planning agent
            evaluator_agent: Policy evaluation and search agent
            output_agent: Results formatting agent
            memory_manager: Optional memory system for few-shot learning
            logger: Optional logger instance
        """
        self.planner = planner_agent
        self.evaluator = evaluator_agent
        self.output = output_agent
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete agent workflow.

        Args:
            initial_state: Dictionary containing:
                - env: Environment configuration
                - workflow: Task workflow DAG
                - params: Cost model parameters

        Returns:
            Final state with optimal policy and formatted output
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING AGENT ORCHESTRATION")
        self.logger.info("=" * 60)

        # Validate input
        self._validate_state(initial_state)

        # Stage 1: Planning
        self.logger.info("\n[Stage 1/3] Invoking Planner Agent...")
        state = self.planner.run(initial_state)

        # Stage 2: Evaluation
        self.logger.info("\n[Stage 2/3] Invoking Evaluator Agent...")
        state = self.evaluator.run(state)

        # Stage 3: Output Generation
        self.logger.info("\n[Stage 3/3] Invoking Output Agent...")
        final_state = self.output.run(state)

        # Store in memory if available
        if self.memory_manager:
            self._store_execution(final_state)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("ORCHESTRATION COMPLETE")
        self.logger.info("=" * 60 + "\n")

        return final_state

    def _validate_state(self, state: Dict[str, Any]) -> None:
        """Validate that required keys are present in state."""
        required_keys = ["env", "workflow", "params"]
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise ValueError(f"Missing required state keys: {missing}")

    def _store_execution(self, final_state: Dict[str, Any]) -> None:
        """Store execution results in memory for future reference."""
        try:
            if hasattr(self.memory_manager, "store_execution"):
                self.memory_manager.store_execution(
                    workflow=final_state.get("workflow"),
                    env=final_state.get("env"),
                    params=final_state.get("params"),
                    policy=final_state.get("optimal_policy"),
                    cost=final_state.get("best_cost"),
                )
                self.logger.info("Execution stored in memory")
        except Exception as e:
            self.logger.warning(f"Failed to store execution in memory: {e}")
