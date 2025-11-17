# Updated version with location indexing starting from 0
# (Full code inserted here and edited accordingly)

#!/usr/bin/env python3
"""
generate_dataset_json_smallvalues_with_id.py

Generate a JSON dataset of workflow DAGs with environment maps
that follow the example pattern you provided.

Each instance JSON object includes:
  - "id": unique UUID string (primary key)
  - "pk": integer primary key (0-based index in file)
  - other fields as before...
"""

import argparse
import json
import math
import random
import uuid
from typing import List, Dict, Any

# =============================
# MAIN CHANGE:
# Locations already range from 0..num_remote.
# This update ensures task->location mappings also start from 0.
# =============================

def make_instance_smallvalues(
    v: int,
    edge_prob: float,
    num_remote: int,
    seed: int
) -> Dict[str, Any]:
    random.seed(seed)

    # simple level-based DAG layout (small DAGs)
    alpha = 1.0
    height = max(1, int(math.ceil(math.sqrt(v) / alpha)))
    width = max(1, int(math.ceil(math.sqrt(v) * alpha)))

    # fill grid with unique task ids 1..v
    grid = [[-1 for _ in range(width)] for _ in range(height)]
    lvlcount = [0] * height
    curr = 1
    for i in range(height):
        grid[i][0] = curr
        lvlcount[i] = 1
        curr += 1
    while curr <= v:
        r = random.randint(0, height - 1)
        if lvlcount[r] >= width:
            continue
        grid[r][lvlcount[r]] = curr
        lvlcount[r] += 1
        curr += 1

    N = v

    # build parent/children temporary maps
    parents = {tid: [] for row in grid for tid in row if tid != -1}
    children = {tid: [] for row in grid for tid in row if tid != -1}

    # mandatory parent connection
    for lvl in range(1, height):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            prev_cnt = lvlcount[lvl-1]
            parent = grid[lvl-1][0] if prev_cnt == 1 else grid[lvl-1][random.randint(0, prev_cnt-1)]
            parents[node].append(parent)
            children[parent].append(node)

    # ensure every non-last-level node has at least one child
    for lvl in range(0, height-1):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            if len(children[node]) == 0:
                nxt_cnt = lvlcount[lvl+1]
                chosen = grid[lvl+1][0] if nxt_cnt == 1 else grid[lvl+1][random.randint(0, nxt_cnt-1)]
                children[node].append(chosen)
                parents[chosen].append(node)

    # extra random edges between consecutive levels
    for lvl in range(0, height-1):
        for i in range(lvlcount[lvl]):
            node = grid[lvl][i]
            for j in range(lvlcount[lvl+1]):
                child = grid[lvl+1][j]
                if (random.random() < edge_prob) and (child not in children[node]):
                    children[node].append(child)
                    parents[child].append(node)

    # collect edges list
    raw_edges = []
    for u, chs in children.items():
        for v_ in chs:
            raw_edges.append((u, v_))
    edgecount = len(raw_edges)

    # --- TASK LOADS (cycles) ---
    tasks: Dict[str, Dict[str, float]] = {}
    for tid in range(1, N+1):
        v_i = random.randint(1_000_000, 35_000_000)
        tasks[str(tid)] = {"v": float(v_i)}

    # --- EDGE SIZES (bytes) ---
    edges_list: List[List[float]] = []
    for (u, v_) in raw_edges:
        size = random.choice([
            random.uniform(0.5e6, 1.5e6),
            random.uniform(1.5e6, 3.0e6),
            random.uniform(3.0e6, 15.0e6)
        ])
        edges_list.append([int(u), int(v_), float(size)])

    # --- LOCATION TYPES ---
    # Already 0..num_remote, so no change needed.
    # enforce that index 0 is always type 0
    tmp_loc = {str(tid - 1): random.randint(0, num_remote) for tid in range(1, N+1)}
    tmp_loc["0"] = 0
    location_types = tmp_loc

    # --- ENV maps follow 0-based indexing ---
    locations = list(range(0, num_remote + 1))
    DR_rows: List[List[float]] = []
    DE_rows: List[List[float]] = []
    VR_rows: List[List[float]] = []
    VE_rows: List[List[float]] = []

    for a in locations:
        for b in locations:
            if a == b:
                DR_rows.append([a, b, 0.0])
            else:
                if a == 0 or b == 0:
                    if random.random() < 0.12:
                        dr_val = random.uniform(1.5e-3, 3.0e-3)
                    else:
                        dr_val = random.uniform(0.8e-5, 2.0e-5)
                else:
                    dr_val = random.uniform(3.0e-5, 6.0e-5)
                DR_rows.append([a, b, float(dr_val)])

    # DE (mJ/byte)
    for l in locations:
        if l == 0:
            de_val = 1.20e-4
        else:
            de_val = random.uniform(1.8e-5, 2.5e-5)
        DE_rows.append([l, float(de_val)])

    # VR (ms/cycle)
    for l in locations:
        if l == 0:
            vr_val = 1.0e-7
        else:
            vr_val = random.uniform(1.0e-8, 4.0e-8)
        VR_rows.append([l, float(vr_val)])

    # VE (mJ/cycle)
    for l in locations:
        if l == 0:
            ve_val = 6.0e-7
        else:
            ve_val = random.uniform(1.2e-7, 3.0e-7)
        VE_rows.append([l, float(ve_val)])

    # costs and mode
    costs = {"CT": 0.2, "CE": 1.20}
    mode = {"delta_t": 1, "delta_e": 1}

    meta = {
        "seed": seed,
        "v": v,
        "edge_prob": edge_prob,
        "num_remote": num_remote,
        "edgecount": edgecount
    }

    instance = {
        "workflow": {"tasks": tasks, "edges": edges_list, "N": N},
        "location_types": location_types,
        "env": {"DR": DR_rows, "DE": DE_rows, "VR": VR_rows, "VE": VE_rows},
        "costs": costs,
        "mode": mode,
        "meta": meta
    }
    return instance


def generate_dataset(
    out_file: str,
    count: int,
    min_v: int,
    max_v: int,
    edge_prob: float,
    num_remote: int,
    seed: int
):
    random.seed(seed)
    dataset = []
    for i in range(count):
        v = random.randint(min_v, max_v)
        s = seed + i
        inst = make_instance_smallvalues(v=v, edge_prob=edge_prob, num_remote=num_remote, seed=s)
        # add primary keys
        inst_id = str(uuid.uuid4())
        inst_pk = i
        inst["id"] = inst_id
        inst["pk"] = inst_pk
        # also add to meta
        inst["meta"]["id"] = inst_id
        inst["meta"]["pk"] = inst_pk

        dataset.append(inst)
    with open(out_file, "w") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"Wrote {len(dataset)} instances to {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="dataset.json")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--min_v", type=int, default=6)
    p.add_argument("--max_v", type=int, default=12)
    p.add_argument("--edge_prob", type=float, default=0.25)
    p.add_argument("--num_remote", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate_dataset(
        out_file=args.out,
        count=args.count,
        min_v=args.min_v,
        max_v=args.max_v,
        edge_prob=args.edge_prob,
        num_remote=args.num_remote,
        seed=args.seed
    )
