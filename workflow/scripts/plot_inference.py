import pandas as pd
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

col = snakemake.params.get('col', None)
row = snakemake.params.get('row', None)

sns.set_theme()

plot = sns.relplot(
    data=df_data, kind="line", x="noise", y="correct_percent", hue='cell_len', 
    height=2.2, aspect=1.61803,
    style='method', palette='tab10', col=col, row=row
)


plot.set_axis_labels(r'edge noise ($\sigma_n$)', r'correct')

plot.savefig(snakemake.output[0])