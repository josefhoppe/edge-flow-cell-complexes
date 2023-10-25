"""
Evaluates the inference accuracy with the given parameters.
"""

from script_utils import method_map

import time
import pandas as pd
from cell_complexes import CellComplexDetectionExperiment
from cell_complexes.generator import TriangulationComplexGenerator, SmallWorldComplexGenerator
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
opt = snakemake.params.get('opt', False)
exp_name = snakemake.params.get('exp_name', '-')


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
    always_opt_cell=opt,
    search_method=method_map[method],
    n_clusters=n_clusters,
)


experiment.run()

df_inference = pd.DataFrame([
        {'model': model, 'size': size, 'flows': flows, 'noise': noise, 'cell_len': cell_len,
         'cells': cells, 'iterations': iterations, 'cell_candidates': cell_candidates, 'n_clusters': n_clusters,
         'run': run, 'correct_abs': correct, 'correct_percent': correct / cells,
         'exp_name': exp_name, 'method': method} 
        for run, correct in enumerate((experiment.correct_cells_found)[flows])
    ]) 

df_inference.to_csv(snakemake.output[0])