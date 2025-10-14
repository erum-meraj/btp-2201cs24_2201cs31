# agentic_offloading/core/network.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class Node:
    """Represents a compute/network node (IoT, Edge, Cloud)."""
    node_id: int
    node_type: str  # 'iot' | 'edge' | 'cloud'
    compute_power: float  # e.g., cycles per second or a relative speed
    energy_coeff: float   # energy cost per unit compute (e.g., mJ per cycle)
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, type={self.node_type}, power={self.compute_power})"


@dataclass
class Link:
    """Represents a directed link between two nodes."""
    src: int
    dst: int
    bandwidth: float  # bytes / second
    delay: float      # fixed delay component in seconds (propagation + processing)
    loss: float = 0.0  # optional packet loss ratio

    def time_per_byte(self) -> float:
        """
        Return time (seconds) to transfer one byte over this link (excluding fixed delay).
        If bandwidth is zero, returns large number.
        """
        if self.bandwidth <= 0:
            return float('inf')
        return 1.0 / self.bandwidth

    def __repr__(self) -> str:
        return f"Link({self.src}->{self.dst} bw={self.bandwidth} delay={self.delay})"


class Network:
    """
    Container for nodes and directed links. Small helper methods to query
    link characteristics used by the environment/cost model.
    """
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        # keyed by (src,dst)
        self.links: Dict[Tuple[int, int], Link] = {}

    # ---------- node operations ----------
    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node id {node.node_id} already exists.")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: int) -> None:
        self.nodes.pop(node_id, None)
        # remove links involving node
        self.links = {k: v for k, v in self.links.items() if k[0] != node_id and k[1] != node_id}

    # ---------- link operations ----------
    def add_link(self, src: int, dst: int, bandwidth: float, delay: float, loss: float = 0.0) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must be added to network before adding a link.")
        self.links[(src, dst)] = Link(src, dst, bandwidth, delay, loss)

    def get_link(self, src: int, dst: int) -> Optional[Link]:
        return self.links.get((src, dst))

    def link_exists(self, src: int, dst: int) -> bool:
        return (src, dst) in self.links

    # ---------- utilities ----------
    def neighbors(self, node_id: int) -> Dict[int, Link]:
        """Return mapping dst -> Link for outgoing edges from node_id."""
        return {dst: link for (s, dst), link in self.links.items() if s == node_id}

    def summary(self) -> str:
        lines = ["Network Summary:"]
        for n in self.nodes.values():
            lines.append(f"  {n}")
        lines.append("Links:")
        for link in self.links.values():
            lines.append(f"  {link}")
        return "\n".join(lines)
