# agents/planner_agent.py
"""
PlannerAgent:
 - takes a Workflow and Environment and produces one or more candidate placement vectors.
 - default planner = greedy topological heuristic (walk tasks in topo order, pick node minimizing local estimate).
 - optional llm_hook: a callable(prompt_str) -> string that can be used to override or augment a candidate.
"""

from typing import List, Callable, Optional, Dict, Tuple
from core.workflow import Workflow, Task
from core.environment import Environment
from core.utils import topological_sort

class PlannerAgent:
    def __init__(self, llm_hook: Optional[Callable[[str], str]] = None):
        """
        llm_hook: optional function that accepts a string prompt and returns a string answer.
                  If provided, PlannerAgent will call it once to request a short strategy
                  and (optionally) a JSON placement suggestion which will be merged into candidates.
    
        """
        self.llm_hook = llm_hook

    def greedy_topological(self, workflow: Workflow, env: Environment) -> List[int]:
        """
        Greedy heuristic:
         - process tasks in topological order
         - for each task, evaluate candidate nodes (all env.network.nodes)
           and pick node minimizing local estimate:
             est_time = exec_time + transfer_time_from_parents
             where exec_time = task.size * VR[node]
                   transfer_time_from_parents = sum_{p in parents} data_p->task * DR(pair(parent_node, node))
         - when parent nodes not yet assigned, assume parents were placed locally (0) or use previously assigned.
        Returns: placement list of length workflow.n_tasks (task index -> node_id)

        """
        # build basic structures
        adj = workflow.adjacency()
        order = topological_sort(adj)
        if order is None:
            raise ValueError("Workflow contains cycle")

        # prepare DR_pair, VR from environment
        DR = env.DR_pair if hasattr(env, "DR_pair") else getattr(env, "DR", {})
        VR = env.VR

        nodes = list(env.network.nodes.keys())
        placement = { }  # task_id -> node_id

        for tid in order:
            task = workflow.tasks[tid]
            # entry/exit tasks: keep local (prefer IoT / node 0 if exists)
            if task.task_id == 0:
                placement[tid] = 0 if 0 in nodes else nodes[0]
                continue

            best_node = None
            best_score = float('inf')
            for node in nodes:
                # execution time estimate
                vr = VR.get(node, float('inf'))
                exec_time = task.size * vr
                # transfer from parents
                transfer_time = 0.0
                for p, data_bytes in task.dependencies.items():
                    p_node = placement.get(p, 0)  # if parent not assigned yet assume 0
                    dr = DR.get((p_node, node), float('inf'))
                    if dr == float('inf'):
                        transfer_time = float('inf')
                        break
                    transfer_time += data_bytes * dr
                score = exec_time + transfer_time
                if score < best_score:
                    best_score = score
                    best_node = node

            # fallback: if nothing found, assign local 0 or first node
            if best_node is None:
                best_node = 0 if 0 in nodes else nodes[0]
            placement[tid] = best_node

        # convert to list ordered by task index
        placement_list = [placement[i] for i in range(len(workflow.tasks))]
        return placement_list

    def generate_candidates(self, workflow: Workflow, env: Environment, num_candidates: int = 3) -> List[List[int]]:
        """
        Generate a small list of candidate placements:
         - greedy baseline
         - greedy + small perturbations (move a few largest tasks to alternative nodes)
         - optional llm candidate if llm_hook provided and returns parseable JSON/list
        """
        candidates: List[List[int]] = []

        greedy = self.greedy_topological(workflow, env)
        candidates.append(greedy)

        # simple perturbations: take top-K largest tasks and try alternative nodes
        nodes = list(env.network.nodes.keys())
        # identify largest tasks (by size)
        tasks_sorted = sorted(enumerate(workflow.tasks), key=lambda x: -x[1].size)
        for perturb_id in range(1, num_candidates):
            cand = greedy.copy()
            # perturb up to perturb_id tasks
            for j, (tid, task) in enumerate(tasks_sorted[:perturb_id]):
                # pick next node cyclically
                current = cand[tid]
                if len(nodes) <= 1:
                    continue
                try:
                    idx = nodes.index(current)
                except ValueError:
                    idx = 0
                alt = nodes[(idx + 1 + j) % len(nodes)]
                cand[tid] = alt
            candidates.append(cand)

        # LLM hook: ask for a suggested placement (best-effort)
        if self.llm_hook is not None:
            prompt = self._build_llm_prompt(workflow, env, examples=candidates[:2])
            try:
                resp = self.llm_hook(prompt)
                # attempt to parse a simple Python/JSON list from the response
                import json, re
                match = re.search(r'\[.*\]', resp, re.S)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list) and len(parsed) == len(workflow.tasks):
                        candidates.append(parsed)
            except Exception:
                pass

        # deduplicate while preserving order
        uniq: List[Tuple[int,...]] = []
        final: List[List[int]] = []
        for c in candidates:
            key = tuple(c)
            if key not in uniq:
                uniq.append(key)
                final.append(c)
        return final

    def _build_llm_prompt(self, workflow: Workflow, env: Environment, examples: Optional[List[List[int]]] = None) -> str:
        # minimal structured prompt — caller may replace with richer prompts
        data = {
            "n_tasks": workflow.n_tasks,
            "tasks": [{ "id": t.task_id, "size": t.size, "parents": t.dependencies } for t in workflow.tasks[:8]],
            "nodes": list(env.network.nodes.keys())
        }
        s = "Given a workflow and nodes, return a JSON array (list) of node-ids (one per task index)\n"
        s += "Context:\n"
        import json
        s += json.dumps(data, indent=2)
        if examples:
            s += "\nExamples (candidate placements):\n"
            for ex in examples:
                s += str(ex) + "\n"
        s += "\nReturn only a JSON list like: [0,1,1,2,...]\n"
        return s
