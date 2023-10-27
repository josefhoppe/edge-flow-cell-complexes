"""
Evaluates the inference accuracy with the given parameters.
"""

from script_utils import method_map

import time
import pandas as pd
from cell_complexes import CellComplexDetectionExperiment
from cell_complexes.generator import TriangulationComplexGenerator, SmallWorldComplexGenerator
from cell_flower.detection import CellCandidateHeuristic
from snakemake.script import Snakemake

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()


method = snakemake.wildcards['method']
model = snakemake.wildcards['model']
size = int(snakemake.wildcards['size'])
flows = int(snakemake.wildcards['flows'])
noise = float(snakemake.wildcards['noise'])
cell_len = int(snakemake.wildcards['cell_len'])
cells = int(snakemake.wildcards['cells'])
iterations = int(snakemake.wildcards['iterations'])
cell_candidates = int(snakemake.wildcards['cell_candidates'])
n_clusters = int(snakemake.wildcards['clusters'])

generator = TriangulationComplexGenerator(
            nodes = size + size // 10,
            delete_nodes = size // 10,
            delete_edges = size // 10,
            cell_lengths = [cell_len] * cells
        )

if model == 'smallworld':
    generator = SmallWorldComplexGenerator(
            nodes=size, 
            cell_lengths = [cell_len] * cells
        )
    
experiment = CellComplexDetectionExperiment(
    cell_compl=generator,
    flow_counts=[flows],
    noise_sigma=noise,
    runs=10,
    cells_per_step=cell_candidates,
    num_curls=iterations,
    approx_epsilon=-1,
    search_method=method_map[method],
    n_clusters=n_clusters
)

experiment.run()

df_approx = experiment.detailed_results

df_approx['model'] = model
df_approx['size'] = size
df_approx['noise'] = noise
df_approx['cell_len'] = cell_len
df_approx['cells'] = cells
df_approx['n_clusters'] = n_clusters

df_approx.to_csv(snakemake.output[0])
