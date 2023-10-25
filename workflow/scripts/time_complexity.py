"""
Tests the runtime for the given parameters
"""

# add code to import path
import sys, resource
from script_utils import method_map

import time
import pandas as pd
from snakemake.script import Snakemake
from cell_complexes import CellComplexDetectionExperiment
from cell_complexes.generator import TriangulationComplexGenerator, SmallWorldComplexGenerator

# too large graphs, will run into recursion limit with spanning tree
class recursionlimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

size: int = int(snakemake.wildcards['size'])
model: str = snakemake.wildcards['model']
method = snakemake.wildcards['method']

generator = TriangulationComplexGenerator(
            nodes = size + size // 10,
            delete_nodes = size // 10,
            delete_edges = size // 10
        )

if model == 'smallworld':
    generator = SmallWorldComplexGenerator(nodes=size, p=0.01)

experiment = CellComplexDetectionExperiment(
    cell_compl=generator,
    flow_counts=[5],
    runs=10,
    search_method=method_map[method],
    n_clusters=9
)

with recursionlimit(20000):
    experiment.run()

df_runtimes = pd.DataFrame([
        {'model': model, 'size': size, 'run': run, 'time': duration, 'method': method, 'n_clusters': 9} 
        for run, duration in enumerate((experiment.execution_time)[5])
    ])

df_runtimes.to_csv(snakemake.output[0])