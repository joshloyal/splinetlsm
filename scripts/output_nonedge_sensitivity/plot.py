import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


gammas = [1.0, 2.0, 3.0, 4.0, 5.0]
n_nodes = 250
n_time_steps = 100
densities = [0.05, 0.1, 0.2, 0.3]

data = []
for density in densities:
    for gamma in gammas:
        res_dir = f'sim_n{n_nodes}_T{n_time_steps}_d{density}_g{gamma}'
        df = pd.read_csv(join(res_dir, 'results.csv'))
        df['n_nodes'] = n_nodes
        df['Expected Density'] = str(density)
        df['n_time_steps'] = n_time_steps
        df['gamma'] = gamma
        data.append(df)


data = pd.concat(data)


fig, ax = plt.subplots(figsize=(8, 6))

sns.lineplot(x='gamma', y='theta_rmse', hue='Expected Density', style='Expected Density', data = data, marker='o', ax=ax, errorbar='sd')
sns.move_legend(ax, "upper left")
ax.set_ylabel('Log-Odds RMSE', fontsize=16)
ax.set_xlabel('Non-Edge to Edge Ratio ($\gamma_n$)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xticks(gammas)
ax.set_ylim([0.1, 0.48])
plt.setp(ax.get_legend().get_title(), fontsize=14)
plt.setp(ax.get_legend().get_texts(), fontsize=14)

fig.savefig('nonedge_sensitivity.pdf', dpi=300, bbox_inches='tight')
