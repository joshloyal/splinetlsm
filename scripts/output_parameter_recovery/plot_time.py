import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


n_nodes = 250
time_steps = [50, 100, 250, 500]
densities = [0.1, 0.2, 0.3]

data = []
for density in densities:
    for n_time_steps in time_steps:
        res_dir = f'gp_n{n_nodes}_T{n_time_steps}_d{density}'
        df = pd.read_csv(join(res_dir, 'results.csv'))
        df['n_nodes'] = n_nodes
        df['Expected Density'] = str(density)
        df['n_time_steps'] = n_time_steps
        data.append(df)


data = pd.concat(data)
fig, ax = plt.subplots(figsize=(15, 3), ncols=3, sharey=True)
fontsize = 12
titlesize = 14

sns.lineplot(x='n_time_steps', y='theta_rmse', hue='Expected Density', style='Expected Density', data = data, marker='o', ax=ax[0],
        errorbar='sd')
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel('Number of Time Steps ($M$)', fontsize=fontsize)
ax[0].set_title('Log-Odds [$\Theta(t)$]', fontsize=titlesize)

sns.lineplot(x='n_time_steps', y='total_coefs_rmse', hue='Expected Density', style='Expected Density', data = data, marker='o', ax=ax[1],
        errorbar='sd')
ax[1].set_xlabel('Number of Time Steps ($M$)', fontsize=fontsize)
#ax[1].set_title('$n = 250$\nCoefficients [$\\beta(t)$]', fontsize=titlesize)
ax[1].set_title('Coefficients [$\\beta(t)$]', fontsize=titlesize)

sns.lineplot(x='n_time_steps', y='U_rmse', hue='Expected Density', style='Expected Density', data = data, marker='o', ax=ax[2],
        errorbar='sd')
ax[2].set_xlabel('Number of Time Steps ($M$)', fontsize=fontsize)
ax[2].set_title('Latent Positions [$U(t)$]', fontsize=titlesize)

for a in ax:
    plt.setp(a.get_legend().get_title(), fontsize=fontsize)
    plt.setp(a.get_legend().get_texts(), fontsize=fontsize)
    a.tick_params(axis='both', which='major', labelsize=fontsize)

fig.savefig('recovery_time.pdf', dpi=300, bbox_inches='tight')
