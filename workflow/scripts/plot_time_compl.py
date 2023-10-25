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

sns.set_theme()

plot = sns.relplot(
    data=df_data, kind="line", x="size", y="time", hue='model', marker='o',
    height=2.2, aspect=1.61803, style='method'
)

plot.set(xscale='log')
plot.set(xlabel=r'size ($|\mathscr{C}_0|$)')
plot.set(yscale='log')
plot.set(ylabel='time [s]')

plot.savefig(snakemake.output[0])