from __future__ import annotations
from typing import Dict, List, Tuple
import math

from core.workflow import Workflow
from core.utils import topological_sort


class UtilityEvaluator:
    """
    Compute offloading cost U(w, p) = delta_t * T + delta_e * E
    where:
      - E = CE * (ED + EV)
      - T = CT * Delta_max(delay-DAG)
    """

    def __init__(self, CT: float = 0.2, CE: float = 1.34, delta_t: int = 1, delta_e: int = 1):
        self.CT = CT
        self.CE = CE
        self.delta_t = delta_t
        self.delta_e = delta_e

    # ---------------- energy computations ----------------
    def compute_ED(self, workflow: Workflow, placement: List[int], DE: Dict[int, float]) -> float:
        """
        ED: total data communication energy cost = sum over tasks of
            DE(li) * (sum_j dj,i + sum_k di,k)
        We'll compute contribution per task i: DE(li) * total_bytes_transferred
        """
        ED = 0.0
        # prepare reverse mapping: for task i, get sum incoming and outgoing data
        n = len(workflow.tasks)
        for i, task in enumerate(workflow.tasks):
            li = placement[i]
            # incoming from parents: sum of data sizes in task.dependencies (parents -> i)
            incoming = sum(task.dependencies.values())
            # outgoing to children: sum of data sizes where task is parent of other tasks
            outgoing = 0.0
            # iterate children by scanning tasks (could be optimized)
            for child in workflow.tasks:
                if i in child.dependencies:
                    outgoing += child.dependencies[i]
            total_bytes = incoming + outgoing
            ED += DE.get(li, 0.0) * total_bytes
        return ED

    def compute_EV(self, workflow: Workflow, placement: List[int], VE: Dict[int, float]) -> float:
        """
        EV: total execution energy = sum_i vi * VE(li)
        """
        EV = 0.0
        for i, task in enumerate(workflow.tasks):
            li = placement[i]
            EV += task.size * VE.get(li, 0.0)
        return EV

    def compute_energy_cost(self, workflow: Workflow, placement: List[int], DE: Dict[int, float], VE: Dict[int, float]) -> float:
        ED = self.compute_ED(workflow, placement, DE)
        EV = self.compute_EV(workflow, placement, VE)
        return self.CE * (ED + EV)

    # ---------------- time computations ----------------
    def compute_delay_edge_weight(self, i: int, j: int, workflow: Workflow,
                                  placement: List[int], DR_pair: Dict[Tuple[int, int], float],
                                  VR: Dict[int, float]) -> float:
        """
        DΔ(i, j) = di,j * DR(li, lj) + vi * VR(li)
        where di,j is data from i -> j (task i produces data di,j consumed by j),
        vi is the task size for task i, li = placement[i], lj = placement[j]
        """
        di_j = 0.0
        # find data size di,j if exists in child's (j) dependencies (parent->child stored in child)
        child = workflow.tasks[j]
        if i in child.dependencies:
            di_j = child.dependencies[i]
        li = placement[i]
        lj = placement[j]
        dr = DR_pair.get((li, lj), float('inf'))
        vr = VR.get(li, 0.0)
        if math.isinf(dr):
            # unreachable -> very large penalty
            return float('inf')
        return di_j * dr + workflow.tasks[i].size * vr

    def compute_critical_path_delay(self, workflow: Workflow, placement: List[int],
                                    DR_pair: Dict[Tuple[int, int], float], VR: Dict[int, float]) -> float:
        """
        Build a weighted DAG using DΔ(i,j) on edges and compute the longest path
        from the entry node (assumed id=0) to the exit node (assumed last id).
        We use topological order and a DP for longest path in DAG (works for DAGs).
        If any edge is infinite (unreachable link), it will propagate to inf.
        """
        adj = workflow.adjacency()
        order = topological_sort(adj)
        if order is None:
            raise ValueError("Workflow must be a DAG (no cycles).")

        # initialize distances to -inf except entry (0)
        dist = {tid: float('-inf') for tid in adj.keys()}
        entry = 0
        exit_id = max(adj.keys())
        dist[entry] = 0.0

        for u in order:
            if dist[u] == float('-inf'):
                # unreachable so skip
                continue
            children = adj.get(u, [])
            for v in children:
                w_uv = self.compute_delay_edge_weight(u, v, workflow, placement, DR_pair, VR)
                if math.isinf(w_uv):
                    dist[v] = float('inf')
                else:
                    if dist[u] + w_uv > dist[v]:
                        dist[v] = dist[u] + w_uv
                # if propagation of inf already happened, it remains
                if math.isinf(dist[v]):
                    # once infinite, stays infinite
                    pass

        return dist[exit_id]

    def compute_time_cost(self, workflow: Workflow, placement: List[int], DR_pair: Dict[Tuple[int, int], float], VR: Dict[int, float]) -> float:
        delta_max = self.compute_critical_path_delay(workflow, placement, DR_pair, VR)
        if math.isinf(delta_max):
            return float('inf')
        return self.CT * delta_max

    # ---------------- total ----------------
    def total_offloading_cost(self, workflow: Workflow, placement: List[int], params: Dict) -> float:
        """
        params: dict with keys 'DR', 'DE', 'VR', 'VE' where:
          - DR is dict mapping (li,lj) -> seconds/byte
          - DE is dict mapping li -> energy/byte
          - VR is dict mapping li -> seconds/cycle
          - VE is dict mapping li -> energy/cycle
        """
        DR_pair = params['DR']
        DE = params['DE']
        VR = params['VR']
        VE = params['VE']

        energy = self.compute_energy_cost(workflow, placement, DE, VE)
        time_cost = self.compute_time_cost(workflow, placement, DR_pair, VR)
        # if any component infinite, return inf
        if math.isinf(energy) or math.isinf(time_cost):
            return float('inf')
        return self.delta_t * time_cost + self.delta_e * energy
