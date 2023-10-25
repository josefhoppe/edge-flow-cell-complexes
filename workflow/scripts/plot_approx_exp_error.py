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

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])

method_dict = {
    'CellSearchMethod.TRIANGLES': 'triangles',
    'CellSearchMethod.MAX': 'max',
    'CellSearchMethod.GROUND_TRUTH': 'true_cells',
    'CellSearchMethod.CLUSTER': 'similarity'
}


df_data['method'] = df_data['method'].map(method_dict.get)
df_data['approx_error'] = np.sqrt(df_data.approx_error)
print(df_data.method.unique())

if 'max_entries' in snakemake.params.keys():
    df_data = df_data[df_data.total_cell_len <= int(snakemake.params['max_entries'])]

if 'sparsity' in snakemake.params.keys():
    sparsity_measure = snakemake.params.get('sparsity_measure', 'total_cell_len')
    df_data = df_data[df_data[sparsity_measure] <= int(snakemake.params.sparsity)].sort_values(sparsity_measure, ascending=False).groupby(['run', 'noise', 'method']).first()

#for run in df_data['run'].unique():
#    max_error = df_data.loc[df_data.run == run, 'approx_error'].max()
#    df_data.loc[df_data.run == run, 'approx_error'] /= max_error

sns.set_theme()

x_axis = snakemake.params.get('x_axis', 'total_cell_len')
x_axis = snakemake.params.get('x_axis', 'total_cell_len')
x_label = snakemake.params.get('x_label', x_axis)

plot: sns.FacetGrid = sns.relplot(
    data=df_data, kind="line", x=x_axis, y="approx_error", hue='method', #marker='o',
    height=2.2, aspect=1.61803,
    units='run', estimator=None, linewidth=.3, hue_order=['triangles', 'max', 'similarity', 'true_cells']
)

if x_label == '$||B_2||_0$':
    x_label = r'$\|\mathbf{B}_2\|_0$'

if x_label == 'scr-C_2':
    x_label = r'$|\mathscr{C}_2|$'

plot.set_axis_labels(x_label, r'$\mathrm{loss}(\mathscr{C},\mathbf{F})$')

plot.savefig(snakemake.output[0])