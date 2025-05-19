import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


output_dir = 'output'
node_sizes = [100, 250, 500, 1000]
degrees = ['log', '0.25', '0.5']
n_time_steps = 100

degree_map = {
    'log': '$O(\log n)$',
    '0.25': '$O(n^{3/4})$',
    '0.5': '$O(n^{1/2})$',
}

data = []
for n_nodes in node_sizes:
    for degree in degrees:
        res_dir = f'gp_n{n_nodes}_T{n_time_steps}_e{degree}'
        df = pd.read_csv(join(output_dir, res_dir, 'results.csv'))
        df['n_nodes'] = n_nodes
        df['n_time_steps'] = n_time_steps
        df['Expected Degree'] = degree_map[degree]
        print(f"Dimension Recovery ({n_nodes}, {degree}): {(df['n_features'] == 2).mean()}")
        data.append(df)


data = pd.concat(data)
fontsize = 12
titlesize = 14

fig, ax = plt.subplots(figsize=(11, 8), nrows=2, ncols=2, sharey='row')
ax = ax.ravel()

sns.lineplot(x='n_nodes', y='theta_rmse',  hue='Expected Degree', style='Expected Degree', data = data, marker='o', ax=ax[0],
        errorbar='sd', hue_order = ['$O(\log n)$', '$O(n^{1/2})$', '$O(n^{3/4})$'])
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel('', fontsize=fontsize)
ax[0].set_title('Log-Odds [$\Theta(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[0].get_xaxis().set_ticklabels([])

sns.lineplot(x='n_nodes', y='coefs_rmse', hue='Expected Degree', style='Expected Degree', data = data, marker='o', ax=ax[2],
        errorbar='sd', hue_order = ['$O(\log n)$', '$O(n^{1/2})$', '$O(n^{3/4})$'])
ax[2].set_ylabel('RMSE', fontsize=fontsize)
ax[2].set_xlabel('Number of Nodes ($n$)', fontsize=fontsize)
ax[2].set_title(r'Coefficients [$\beta_1(t), \beta_2(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)


sns.lineplot(x='n_nodes', y='U_rmse_select', hue='Expected Degree', style='Expected Degree', data = data, marker='o', ax=ax[1],
        errorbar='sd', hue_order = ['$O(\log n)$', '$O(n^{1/2})$', '$O(n^{3/4})$'])
ax[1].set_xlabel('', fontsize=fontsize)
ax[1].set_title('Latent Positions [$U(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].get_xaxis().set_ticklabels([])

sns.lineplot(x='n_nodes', y='intercept_rmse', hue='Expected Degree', style='Expected Degree', data = data, marker='o', ax=ax[3],
        errorbar='sd', hue_order = ['$O(\log n)$', '$O(n^{1/2})$', '$O(n^{3/4})$'])
ax[3].set_xlabel('Number of Nodes ($n$)', fontsize=fontsize)
ax[3].set_title(r'Intercept [$\rho_n(t)$]', fontsize=titlesize)
ax[3].tick_params(axis='both', which='major', labelsize=fontsize)

for a in ax:
    plt.setp(a.get_legend().get_title(), fontsize=fontsize)
    plt.setp(a.get_legend().get_texts(), fontsize=fontsize)

fig.savefig('recovery_nodes.pdf', dpi=300, bbox_inches='tight')
