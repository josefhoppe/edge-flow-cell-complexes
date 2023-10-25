import matplotlib.pyplot as plt
import pandas as pd

def runtime_exp():
    """
    runtime in relation to graph size experiment
    """
    df_runtimes = pd.read_csv('results/runtime_exp.csv', index_col=0)
    df_runtimes.columns = df_runtimes.columns.map(int)
    print(df_runtimes.mean())
    fig, ax = plt.subplots()
    ax.fill_between(df_runtimes.columns, df_runtimes.mean() - df_runtimes.std(), df_runtimes.mean() + df_runtimes.std(), color='lightblue')
    ax.plot(df_runtimes.mean(), linestyle='-', marker='o')
    ax.set_xlabel('number of nodes in graph [-]')
    ax.set_ylabel('time [s]')
    ax.loglog()
    return fig

figures = {
    'time_exp': runtime_exp()
}

for name, fig in figures.items():
    fig.savefig(f'paper/figures/{name}.pdf')