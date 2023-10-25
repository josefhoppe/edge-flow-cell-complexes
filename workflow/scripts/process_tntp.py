"""
Processes the taxi trajectory data to a graph with edge flows
"""

# add code to import path
from pathlib import Path
import sys, resource
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import time
import pandas as pd
from ast import literal_eval
from snakemake.script import Snakemake

def import_trips(matfile: str, min_flow_ratio = 0.01) -> tuple[list[tuple], pd.DataFrame]:
    """
    imports a matrix of flows from the TransportationNetworks format.

    The result is a tuple (cell list, flow matrix)
    - cell list: list of all nodes as singleton tuples followed by all edges. Can be used to construct CellComplex
    - flow matrix: pandas DataFrame with edges as columns and rows representing the flows (currently always one row)

    Adapted from https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb
    """
    f = open(matfile, 'r')
    all_rows = f.read()
    f.close()
    
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for block in blocks:
        orig = block.split('\n')
        dests = orig[1:]
        orig=int(orig[0])

        d = [literal_eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations

    node_set = set()
    edge_set = set()
    flow_map = {}
    for u, targets in matrix.items():
        node_set.add(u-1)
        for v, flow in targets.items():
            edge_set.add((min(u,v) - 1, max(u,v) -1))
            flow_map[(u-1,v-1)] = flow
            node_set.add(v-1)
    nodes = [(n,) for n in range(max(node_set) + 1)]

    edges = sorted(edge_set)
    
    res = pd.Series(.0, index=edges)
    for (a,b) in edges:
        res[(a,b)] = flow_map.get((a,b),0) - flow_map.get((b,a),0)
    
    res = res[res.abs() >= min_flow_ratio * res.abs().max()]
    edges = sorted(res.index.to_list())

    return nodes + edges, pd.DataFrame(res).T

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

tripfile = snakemake.input[0]

cells, df_flows = import_trips(tripfile)

max_flow = df_flows.abs().max().max()
total_flow = df_flows.abs().sum().sum()

print(f"processed flow, max={max_flow:.2f}, total={total_flow:.2f}, edges={len(df_flows.columns)}, nodes={len([c for c in cells if len(c) == 1])}, flows={len(df_flows.index)}")

print((df_flows > 1).sum().sum())

graph_file = open(snakemake.output[0], 'w')
graph_file.write(str(cells))
graph_file.close()

df_flows.to_csv(snakemake.output[1])