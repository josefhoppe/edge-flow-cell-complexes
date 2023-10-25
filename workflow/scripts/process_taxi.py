"""
Processes the taxi trajectory data to a graph with edge flows
"""

# add code to import path
from pathlib import Path
import sys, resource
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import time
import pandas as pd
from cell_complexes import CellComplexDetectionExperiment
from cell_complexes.generator import TriangulationComplexGenerator, SmallWorldComplexGenerator
from cell_flower.detection import CellSearchMethod
import itertools
from collections import defaultdict
from snakemake.script import Snakemake

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

trajectory_file = open(snakemake.input[0], 'r')

all_flows = []
all_nodes = set()
all_edges = set()

line: str
for line in trajectory_file.readlines():
    trajectory = list(map(int,line.split(',')))
    flow_dict = defaultdict(lambda: 0)
    all_flows.append(flow_dict)
    for a,b in itertools.pairwise(trajectory):
        all_nodes.add(a)
        all_nodes.add(b)
        if a < b:
            if a != 0 and b != 0:
                flow_dict[(a,b)] += 1
            all_edges.add((a,b))
        if a > b:
            if a != 0 and b != 0:
                flow_dict[(b,a)] -= 1
            all_edges.add((b,a))

edges = sorted(all_edges)

cells = [(x,) for x in sorted(all_nodes)] + edges

graph_file = open(snakemake.output[0], 'w')
graph_file.write(str(cells))
graph_file.close()

df_flows = pd.DataFrame(all_flows, columns=edges)
df_flows.to_csv(snakemake.output[1])