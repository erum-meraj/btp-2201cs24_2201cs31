from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from core.utils import topological_sort


@dataclass
class Task:
    """Represents a task (vertex) in the workflow DAG."""
    task_id: int
    size: float  # CPU cycles required
    dependencies: Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, size={self.size}, deps={list(self.dependencies.keys())})"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "size": self.size,
            "dependencies": self.dependencies
        }

    @staticmethod
    def from_dict(data: dict) -> Task:
        return Task(
            task_id=data["task_id"],
            size=data["size"],
            dependencies=data.get("dependencies", {})
        )


class Workflow:
    """Represents a DAG workflow."""
    def __init__(self, tasks: Optional[List[Task]] = None, entry_exit: bool = True):
        self.entry_exit = entry_exit
        self.tasks: List[Task] = tasks or []
        if entry_exit and len(self.tasks) == 0:
            entry = Task(task_id=0, size=1.0, dependencies={})
            exit_task = Task(task_id=1, size=1.0, dependencies={0: 0.0})
            self.tasks = [entry, exit_task]
        self._normalize_ids()

    def _normalize_ids(self) -> None:
        for i, t in enumerate(self.tasks):
            t.task_id = i

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {t.task_id: [] for t in self.tasks}
        for t in self.tasks:
            for parent in t.dependencies.keys():
                if parent not in adj:
                    adj[parent] = []
                adj[parent].append(t.task_id)
        return adj

    def validate_dag(self) -> bool:
        order = topological_sort(self.adjacency())
        if order is None:
            raise ValueError("Workflow contains a cycle.")
        return True

    # ---------- Serialization ----------
    def to_dict(self) -> dict:
        return {
            "entry_exit": self.entry_exit,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Workflow:
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        wf = cls(tasks=tasks, entry_exit=data.get("entry_exit", True))
        wf._normalize_ids()
        return wf

    def __repr__(self) -> str:
        return f"Workflow(num_tasks={self.n_tasks})"
