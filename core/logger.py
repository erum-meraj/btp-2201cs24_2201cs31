"""
Enhanced Logging System for Agentic Task Offloading

Provides developer-friendly logging with:
- Structured log levels (DEBUG, INFO, WARNING, ERROR)
- Component-based logging (Planner, Evaluator, Tools)
- Formatted output with timestamps
- Optional detailed trace files
- Console output control
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class AgenticLogger:
    """
    Enhanced logger for the multi-agent system.
    
    Features:
    - Component-specific logging (Planner, Evaluator, Tools)
    - Structured JSON logs for trace files
    - Clean console output for developers
    - Automatic log rotation
    """
    
    def __init__(
        self,
        log_file: str = "agent_trace.log",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_trace: bool = True
    ):
        """
        Initialize the logging system.
        
        Args:
            log_file: Path to log file
            console_level: Console logging level (INFO, DEBUG, WARNING)
            file_level: File logging level (always more detailed)
            enable_trace: If True, creates detailed trace files
        """
        self.log_file = log_file
        self.enable_trace = enable_trace
        
        # Create main logger
        self.logger = logging.getLogger('AgenticSystem')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Console handler (clean output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(component)-12s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (detailed output)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(component)-12s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Trace file for detailed interactions
        self.trace_file = None
        if enable_trace:
            trace_path = log_file.replace('.log', '_trace.json')
            self.trace_file = Path(trace_path)
            self.trace_entries = []
    
    def _log(self, level: int, component: str, message: str, **kwargs):
        """Internal logging method with component tracking."""
        extra = {'component': component}
        self.logger.log(level, message, extra=extra, **kwargs)
    
    # Component-specific logging methods
    
    def orchestrator(self, message: str, level: int = logging.INFO):
        """Log from orchestrator."""
        self._log(level, 'Orchestrator', message)
    
    def planner(self, message: str, level: int = logging.INFO):
        """Log from planner agent."""
        self._log(level, 'Planner', message)
    
    def evaluator(self, message: str, level: int = logging.INFO):
        """Log from evaluator agent."""
        self._log(level, 'Evaluator', message)
    
    def output_agent(self, message: str, level: int = logging.INFO):
        """Log from output agent."""
        self._log(level, 'Output', message)
    
    def tool(self, tool_name: str, message: str, level: int = logging.DEBUG):
        """Log from tools (candidate generator, utility, etc.)."""
        self._log(level, f'Tool:{tool_name}', message)
    
    def llm_call(self, agent: str, prompt_preview: str, tokens: Optional[int] = None):
        """Log LLM API call."""
        msg = f"LLM call - Preview: {prompt_preview[:100]}..."
        if tokens:
            msg += f" | Tokens: {tokens}"
        self._log(logging.DEBUG, f'{agent}:LLM', msg)
        
        if self.enable_trace:
            self._add_trace('llm_call', {
                'agent': agent,
                'prompt_preview': prompt_preview[:200],
                'tokens': tokens,
                'timestamp': datetime.now().isoformat()
            })
    
    def llm_response(self, agent: str, response_preview: str, tokens: Optional[int] = None):
        """Log LLM response."""
        msg = f"LLM response - Preview: {response_preview[:100]}..."
        if tokens:
            msg += f" | Tokens: {tokens}"
        self._log(logging.DEBUG, f'{agent}:LLM', msg)
        
        if self.enable_trace:
            self._add_trace('llm_response', {
                'agent': agent,
                'response_preview': response_preview[:200],
                'tokens': tokens,
                'timestamp': datetime.now().isoformat()
            })
    
    def tool_call(self, tool_name: str, action: str, params: Dict[str, Any]):
        """Log tool invocation."""
        param_str = json.dumps(params, default=str)[:100]
        self._log(logging.DEBUG, f'Tool:{tool_name}', f"{action} | Params: {param_str}...")
        
        if self.enable_trace:
            self._add_trace('tool_call', {
                'tool': tool_name,
                'action': action,
                'params': params,
                'timestamp': datetime.now().isoformat()
            })
    
    def tool_result(self, tool_name: str, action: str, result_summary: str):
        """Log tool result."""
        self._log(logging.DEBUG, f'Tool:{tool_name}', f"{action} complete | {result_summary}")
        
        if self.enable_trace:
            self._add_trace('tool_result', {
                'tool': tool_name,
                'action': action,
                'result': result_summary,
                'timestamp': datetime.now().isoformat()
            })
    
    def policy_evaluation(self, policy: tuple, cost: float, is_best: bool = False):
        """Log policy evaluation."""
        status = "NEW BEST" if is_best else "evaluated"
        self._log(
            logging.INFO if is_best else logging.DEBUG,
            'Evaluator',
            f"Policy {policy} {status} | Cost: {cost:.6f}"
        )
    
    def stage_start(self, stage: str, description: str):
        """Log stage start."""
        self._log(logging.INFO, 'Orchestrator', f"{'='*50}")
        self._log(logging.INFO, 'Orchestrator', f"STAGE: {stage}")
        self._log(logging.INFO, 'Orchestrator', f"{description}")
        self._log(logging.INFO, 'Orchestrator', f"{'='*50}")
    
    def stage_complete(self, stage: str, duration: Optional[float] = None):
        """Log stage completion."""
        msg = f"{stage} complete"
        if duration:
            msg += f" | Duration: {duration:.2f}s"
        self._log(logging.INFO, 'Orchestrator', msg)
    
    def experiment_start(self, experiment_id: str, num_tasks: int, num_locations: int):
        """Log experiment start."""
        self._log(logging.INFO, 'Main', f"\n{'='*70}")
        self._log(logging.INFO, 'Main', f"Experiment: {experiment_id}")
        self._log(logging.INFO, 'Main', f"Tasks: {num_tasks} | Locations: {num_locations}")
        self._log(logging.INFO, 'Main', f"{'='*70}")
    
    def experiment_complete(self, experiment_id: str, optimal_policy: list, best_cost: float):
        """Log experiment completion."""
        self._log(logging.INFO, 'Main', f"Experiment {experiment_id} complete")
        self._log(logging.INFO, 'Main', f"Optimal: {optimal_policy} | Cost: {best_cost:.6f}")
        self._log(logging.INFO, 'Main', f"{'='*70}\n")
    
    def error(self, component: str, error: Exception, context: str = ""):
        """Log error with context."""
        msg = f"ERROR: {str(error)}"
        if context:
            msg += f" | Context: {context}"
        self._log(logging.ERROR, component, msg)
        
        if self.enable_trace:
            self._add_trace('error', {
                'component': component,
                'error': str(error),
                'context': context,
                'timestamp': datetime.now().isoformat()
            })
    
    def warning(self, component: str, message: str):
        """Log warning."""
        self._log(logging.WARNING, component, message)
    
    def _add_trace(self, event_type: str, data: Dict[str, Any]):
        """Add entry to trace file."""
        if self.enable_trace:
            self.trace_entries.append({
                'type': event_type,
                'data': data
            })
    
    def flush_trace(self):
        """Write trace entries to file."""
        if self.enable_trace and self.trace_entries and self.trace_file:
            try:
                with open(self.trace_file, 'a', encoding='utf-8') as f:
                    for entry in self.trace_entries:
                        f.write(json.dumps(entry, default=str) + '\n')
                self.trace_entries = []
            except Exception as e:
                self.logger.error(f"Failed to flush trace: {e}")
    
    def close(self):
        """Close logger and flush traces."""
        self.flush_trace()
        for handler in self.logger.handlers:
            handler.close()


# Singleton instance
_logger_instance = None


def get_logger(
    log_file: str = "agent_trace.log",
    console_level: int = logging.INFO,
    reset: bool = False
) -> AgenticLogger:
    """
    Get or create the global logger instance.
    
    Args:
        log_file: Path to log file
        console_level: Console logging level
        reset: If True, create new logger instance
        
    Returns:
        AgenticLogger instance
    """
    global _logger_instance
    
    if _logger_instance is None or reset:
        _logger_instance = AgenticLogger(
            log_file=log_file,
            console_level=console_level
        )
    
    return _logger_instance


def set_log_level(console_level: int = logging.INFO, file_level: int = logging.DEBUG):
    """
    Update logging levels.
    
    Args:
        console_level: Console logging level (INFO, DEBUG, WARNING, ERROR)
        file_level: File logging level
    """
    logger = get_logger()
    for handler in logger.logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(console_level)
        elif isinstance(handler, logging.FileHandler):
            handler.setLevel(file_level)