import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


gammas = [0.1, 0.25, 0.5, 0.8]
n_nodes = 250
time_steps = [50, 100, 250, 500]
density = 0.2

data = []
for n_time_steps in time_steps:
    for gamma in gammas:
        res_dir = f'sim_n{n_nodes}_T{n_time_steps}_d{density}_g{gamma}'
        df = pd.read_csv(join(res_dir, 'results.csv'))
        df['n_nodes'] = n_nodes
        df['density'] = density
        df['Number of Time Steps ($M$)'] = str(n_time_steps)
        df['gamma'] = gamma
        data.append(df)


data = pd.concat(data)

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x='gamma', y='theta_rmse', hue='Number of Time Steps ($M$)', style='Number of Time Steps ($M$)', data = data, marker='o', ax=ax,
        errorbar='sd')
ax.set_ylabel('Log-Odds RMSE', fontsize=16)
ax.set_xlabel('Temporal Snapshot Fraction ($\gamma_M$)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xticks(gammas)
plt.setp(ax.get_legend().get_title(), fontsize=14)
plt.setp(ax.get_legend().get_texts(), fontsize=14)
fig.savefig('snapshot_fraction_sensitivity.pdf', dpi=300, bbox_inches='tight')
