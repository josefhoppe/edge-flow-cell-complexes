from math import isnan
import pandas as pd
import numpy as np
import seaborn as sns
from snakemake.script import Snakemake

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()
x_axis = snakemake.params.get('x_axis', 'total_cell_len')
x_label = snakemake.params.get('x_label', x_axis)

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])
df_data['approx_error'] = np.sqrt(df_data.approx_error)

method_dict = {
    'CellCandidateHeuristic.TRIANGLES': 'triangles',
    'CellCandidateHeuristic.MAX': 'max',
    'CellCandidateHeuristic.GROUND_TRUTH': 'true_cells',
    'CellCandidateHeuristic.SIMILARITY': 'similarity'
}

df_data['method'] = df_data['method'].map(method_dict.get)

if 'max_x' in snakemake.params.keys():
    df_data = df_data[df_data[x_axis] <= int(snakemake.params['max_x'])]
else:
    # find x where first approach is below 1% of initial error
    max_error = df_data.approx_error.max()
    max_x = df_data[df_data.approx_error < 0.01 * max_error][x_axis].min()
    if 'alt_max_x' in snakemake.params.keys():
        alt_max = int(snakemake.params.alt_max_x)
        if np.isnan(max_x) or max_x > alt_max:
            max_x = alt_max
    if not isnan(max_x):
        df_data = df_data[df_data[x_axis] <= max_x]

#for run in df_data['run'].unique():
#    max_error = df_data.loc[df_data.run == run, 'approx_error'].max()
#    df_data.loc[df_data.run == run, 'approx_error'] /= max_error

sns.set_theme()


plot: sns.FacetGrid = sns.relplot(
    data=df_data, kind="line", x=x_axis, y="approx_error", hue='method', #marker='o',
    height=2.2, aspect=1.61803,
    style='cell_candidates'
)

if x_label == '$||B_2||_0$':
    x_label = r'$\|\mathbf{B}_2\|_0$'

plot.set_axis_labels(x_label, r'$\mathrm{loss}(\mathscr{C},\mathbf{F})$')
plot.set(yscale='log')

plot.savefig(snakemake.output[0])