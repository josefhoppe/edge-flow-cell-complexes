"""
Evaluates the inference accuracy with the given parameters.
"""

from script_utils import method_map

import pandas as pd
from cell_complexes import CellComplexDetectionExperiment
from cell_flower.cell_complex import CellComplex
from ast import literal_eval
from snakemake.script import Snakemake

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

method = snakemake.wildcards['method']
flows = int(snakemake.wildcards['flows'])
iterations = int(snakemake.wildcards['iterations'])
cell_candidates = int(snakemake.wildcards['cell_candidates'])
n_clusters = int(snakemake.wildcards['clusters'])

cell_compl_file = open(snakemake.input[0])
realworld_complex = CellComplex(literal_eval(cell_compl_file.read()))

df_flows = pd.read_csv(snakemake.input[1], index_col=0).fillna(0)
df_flows = df_flows.groupby(df_flows.index.map(int) % flows).sum()
    
experiment = CellComplexDetectionExperiment(
    cell_compl=realworld_complex,
    flows=[df_flows.to_numpy()  for _ in range(20)],
    flow_counts=[flows],
    runs=1,
    cells_per_step=cell_candidates,
    num_curls=iterations,
    approx_epsilon=-1,
    search_method=method_map[method],
    n_clusters=n_clusters,
    agg_results=False
)

experiment.run()

df_approx = experiment.detailed_results

df_approx.to_csv(snakemake.output[0])

inf_cells_file = open(snakemake.output[1], 'w')
inf_cells_file.write(str(experiment.inferred_cells[flows]))
inf_cells_file.close()