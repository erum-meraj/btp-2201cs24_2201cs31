from __future__ import annotations
from typing import Dict, Tuple, Optional
import random
import numpy as np
from core.network import Network, Link


class Environment:
    """
    Holds dynamic parameters for the edge-cloud environment:
    - DR(li, lj): time per byte to move data between li and lj (seconds/byte)
    - DE(li): energy per byte for data communication at location li (mJ/byte)
    - VR(li): time per unit compute (seconds per CPU-cycle) at location li
    - VE(li): energy per unit compute (mJ per CPU-cycle) at location li

    The class supports:
    - randomization (sample from ranges)
    - setting parameters directly (useful for reproducible experiments)
    - querying pairwise DR using network links if available
    """
    def __init__(self, network: Network):
        self.network = network
        # By default store node-local params as dict node_id -> value
        self.DE: Dict[int, float] = {}
        self.VR: Dict[int, float] = {}
        self.VE: Dict[int, float] = {}
        # For DR (pairwise) we will compute from the network link if available
        # or fall back to node-local "transfer speed" estimation
        # Optionally a precomputed DR matrix can be set.
        self.DR_pair: Dict[Tuple[int, int], float] = {}

    # ---------- initialization ----------
    def randomize(self,
                  dr_range: Tuple[float, float] = (1e-7, 5e-7),
                  de_range: Tuple[float, float] = (1e-4, 1e-3),
                  vr_range: Tuple[float, float] = (1e-9, 1e-8),
                  ve_range: Tuple[float, float] = (1e-5, 1e-4),
                  seed: Optional[int] = None) -> None:
        """
        Randomize environment parameters.
        Units are intentionally general; typical usages:
        - dr_range: seconds per byte (smaller = faster)
        - de_range: mJ per byte
        - vr_range: seconds per cpu-cycle (smaller = faster)
        - ve_range: mJ per cpu-cycle
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for node_id in self.network.nodes.keys():
            self.DE[node_id] = float(random.uniform(*de_range))
            self.VR[node_id] = float(random.uniform(*vr_range))
            self.VE[node_id] = float(random.uniform(*ve_range))

        # For pairwise DR, prefer to use existing links; otherwise fall back to
        # sum of src/dst time_per_byte + link.delay-per-byte heuristic.
        for src in self.network.nodes:
            for dst in self.network.nodes:
                if src == dst:
                    self.DR_pair[(src, dst)] = 0.0
                    continue
                link = self.network.get_link(src, dst)
                if link:
                    # time per byte = link time_per_byte + fixed delay divided by a normalizing bytes
                    self.DR_pair[(src, dst)] = link.time_per_byte() + (link.delay / 1e6)  # normalize delay/byte
                else:
                    # if no direct link, assume large transfer time to discourage that placement
                    self.DR_pair[(src, dst)] = float('inf')

    # ---------- setters / getters ----------
    def set_DE(self, node_id: int, value: float) -> None:
        self.DE[node_id] = value

    def set_VR(self, node_id: int, value: float) -> None:
        self.VR[node_id] = value

    def set_VE(self, node_id: int, value: float) -> None:
        self.VE[node_id] = value

    def set_DR_pair(self, src: int, dst: int, dr_value: float) -> None:
        self.DR_pair[(src, dst)] = dr_value

    def get_DE(self, node_id: int) -> float:
        return self.DE[node_id]

    def get_VR(self, node_id: int) -> float:
        return self.VR[node_id]

    def get_VE(self, node_id: int) -> float:
        return self.VE[node_id]

    def get_DR(self, src: int, dst: int) -> float:
        """
        Return DR(src, dst) - time per byte to move between src and dst.
        If there is no valid path / link, returns inf.
        """
        return self.DR_pair.get((src, dst), float('inf'))

    def get_all_parameters(self) -> Dict:
        return {
            'DR': self.DR_pair,
            'DE': self.DE,
            'VR': self.VR,
            'VE': self.VE
        }
