from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass, field
from core.utils import topological_sort


@dataclass
class Task:
    """Represents a task (vertex) in the workflow DAG."""
    task_id: int
    size: float  # in CPU cycles required (vi)
    # dependencies: mapping from parent_task_id -> data size (bytes) needed from that parent
    dependencies: Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, size={self.size}, deps={list(self.dependencies.keys())})"


class Workflow:
    """
    Represents a DAG workflow as used in the paper.
    - tasks: ordered list: index corresponds to task index (0..N+1 if entry/exit included)
    - entry (0) and exit (N+1) can be optionally included
    """
    def __init__(self, tasks: Optional[List[Task]] = None, entry_exit: bool = True):
        self.entry_exit = entry_exit
        self.tasks: List[Task] = tasks or []
        if entry_exit:
            # ensure there is an entry (id 0) and exit (id last)
            if len(self.tasks) == 0:
                # create minimal entry and exit
                entry = Task(task_id=0, size=1.0, dependencies={})
                exit_task = Task(task_id=1, size=1.0, dependencies={0: 0.0})
                self.tasks = [entry, exit_task]
        self._normalize_ids()

    def _normalize_ids(self) -> None:
        """Ensure task ids are contiguous indices (0..len-1)."""
        for i, t in enumerate(self.tasks):
            t.task_id = i

    @property
    def n_tasks(self) -> int:
        # exclude entry and exit if desired, but keep representation general
        return len(self.tasks)

    def adjacency(self) -> Dict[int, List[int]]:
        """Return adjacency mapping parent -> list(children)."""
        adj: Dict[int, List[int]] = {t.task_id: [] for t in self.tasks}
        for t in self.tasks:
            for parent in t.dependencies.keys():
                if parent not in adj:
                    adj[parent] = []
                adj[parent].append(t.task_id)
        return adj

    def parents_of(self, task_id: int) -> List[int]:
        return list(self.tasks[task_id].dependencies.keys())

    def add_task(self, size: float, dependencies: Optional[Dict[int, float]] = None) -> Task:
        if dependencies is None:
            dependencies = {}
        new_id = len(self.tasks)
        t = Task(task_id=new_id, size=size, dependencies=dependencies)
        self.tasks.append(t)
        return t

    def validate_dag(self) -> bool:
        """
        Basic validation that DAG has no cycles via topological sort.
        Raises ValueError if cycle detected.
        """
        order = topological_sort(self.adjacency())
        if order is None:
            raise ValueError("Workflow contains a cycle.")
        return True

    # ---------- convenience generator ----------
    @staticmethod
    def random_dag(num_tasks: int,
                   min_size: float = 1e6,
                   max_size: float = 1e7,
                   max_parents: int = 2,
                   data_min: float = 10 * 1024 * 1024,  # bytes
                   data_max: float = 30 * 1024 * 1024  # bytes
                   ) -> Workflow:
        """
        Generate a random DAG with `num_tasks` tasks EXCLUDING entry/exit.
        The method will create an entry node (0) and exit node (N+1) internally.
        """
        if num_tasks < 1:
            raise ValueError("num_tasks must be >= 1 (excluding entry/exit).")

        wf = Workflow(entry_exit=True)
        # entry node is already id 0
        # create interior tasks
        for i in range(num_tasks):
            size = random.uniform(min_size, max_size)
            dependencies = {}
            # random parents chosen from existing tasks (including entry)
            possible_parents = list(range(len(wf.tasks)))
            # ensure at least one parent (entry) for first tasks
            k = random.randint(1, min(max_parents, len(possible_parents)))
            parents = random.sample(possible_parents, k)
            for p in parents:
                data_bytes = random.uniform(data_min, data_max)
                dependencies[p] = data_bytes
            wf.add_task(size=size, dependencies=dependencies)

        # connect last tasks to exit (ensure exit exists)
        # make exit the last node; add dependencies from some of previous tasks
        exit_idx = len(wf.tasks) - 1
        # ensure exit has at least one parent (the last added task(s))
        parents = random.sample(range(1, exit_idx), min(2, max(1, exit_idx - 1)))
        wf.tasks[exit_idx].dependencies = {p: random.uniform(data_min, data_max) for p in parents}
        # normalize task ids
        wf._normalize_ids()
        wf.validate_dag()
        return wf

    def __repr__(self) -> str:
        return f"Workflow(num_tasks={self.n_tasks})"