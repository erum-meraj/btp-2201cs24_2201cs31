"""
Configuration Module

Provides easy configuration and initialization of the agentic system.
"""

import os
from typing import Optional


class AgentConfig:
    """Configuration for the multi-agent system."""

    def __init__(
        self,
        api_key: str,
        log_file: str = "agent_trace.txt",
        model_name: str = "models/gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        """
        Initialize configuration.

        Args:
            api_key: API key for LLM (Google Gemini)
            log_file: Path to log file for agent interactions
            model_name: LLM model to use
            temperature: Temperature for LLM sampling
        """
        self.api_key = api_key
        self.log_file = log_file
        self.model_name = model_name
        self.temperature = temperature

    @classmethod
    def from_env(cls, log_file: str = "agent_trace.txt") -> "AgentConfig":
        """
        Create configuration from environment variables.

        Expected env vars:
        - GOOGLE_API_KEY or GEMINI_API_KEY
        - MODEL_NAME (optional)
        - TEMPERATURE (optional)

        Args:
            log_file: Path to log file

        Returns:
            AgentConfig instance
        """
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable"
            )

        model_name = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")
        temperature = float(os.getenv("TEMPERATURE", "0.3"))

        return cls(
            api_key=api_key,
            log_file=log_file,
            model_name=model_name,
            temperature=temperature,
        )


class SystemBuilder:
    """Builder for creating the complete agent system."""

    def __init__(self, config: AgentConfig):
        """
        Initialize builder with configuration.

        Args:
            config: AgentConfig instance
        """
        self.config = config

    def build(self, memory_manager=None):
        """
        Build and return the complete agent system.

        Args:
            memory_manager: Optional memory manager instance

        Returns:
            Tuple of (orchestrator, planner, evaluator, output)
        """
        # Import required modules with CORRECT PATHS
        from agents.base_agent.base_agent import BaseAgent
        from agents.planner_agent.planner import PlannerAgent
        from agents.output_agent.output import OutputAgent
        from agents.evaluator_agent.evaluator import EvaluatorAgent
        from agents.orchestrator.orchestrator import AgentOrchestrator

        # Assumed core modules (ensure these exist in your path)
        try:
            from core.workflow import Workflow
            from core.environment import Environment
            from core.cost_eval import UtilityEvaluator
        except ImportError:
            # Fallback for testing if core modules aren't available
            print(
                "Warning: Core modules not found. Using mocks/placeholders if available."
            )
            Workflow = None
            Environment = None
            UtilityEvaluator = None

        # Create base agent for LLM access
        base_agent = BaseAgent(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )

        # Create planner agent
        planner = PlannerAgent(
            api_key=self.config.api_key,
            log_file=self.config.log_file,
            memory_manager=memory_manager,
        )

        # Create evaluator agent with tools
        evaluator = EvaluatorAgent(
            base_agent=base_agent,
            workflow_module=Workflow,
            environment_module=Environment,
            cost_evaluator_class=UtilityEvaluator,
            memory_manager=memory_manager,
            log_file=self.config.log_file,
        )

        # Create output agent
        output_agent = OutputAgent(
            api_key=self.config.api_key, log_file=self.config.log_file
        )

        # Create orchestrator
        orchestrator = AgentOrchestrator(
            planner_agent=planner,
            evaluator_agent=evaluator,
            output_agent=output_agent,
            memory_manager=memory_manager,
        )

        return orchestrator, planner, evaluator, output_agent

    def build_orchestrator(self, memory_manager=None):
        """
        Build and return just the orchestrator (most common use case).

        Args:
            memory_manager: Optional memory manager instance

        Returns:
            AgentOrchestrator instance
        """
        orchestrator, _, _, _ = self.build(memory_manager)
        return orchestrator


def create_system(
    api_key: Optional[str] = None,
    log_file: str = "agent_trace.txt",
    memory_manager=None,
):
    """
    Convenience function to create the agent system.

    Args:
        api_key: API key (if None, reads from environment)
        log_file: Path to log file
        memory_manager: Optional memory manager

    Returns:
        AgentOrchestrator instance ready to use
    """
    if api_key:
        config = AgentConfig(api_key=api_key, log_file=log_file)
    else:
        config = AgentConfig.from_env(log_file=log_file)

    # Ensure the directory for the log file exists to avoid FileNotFoundError
    log_dir = os.path.dirname(log_file)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            raise ValueError(f"Unable to create log directory: {log_dir}")

    builder = SystemBuilder(config)
    return builder.build_orchestrator(memory_manager)
