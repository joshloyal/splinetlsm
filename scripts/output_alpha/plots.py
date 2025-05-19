import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


output_dir = 'output'
node_sizes = [200]
n_time_steps = 100
density = 0.2 
alphas = [0.8, 0.85, 0.9, 0.95, 0.99]

data = []
for n_nodes in node_sizes:
    for alpha in alphas:
        res_dir = f'a{alpha}_n{n_nodes}_T{n_time_steps}_d{density}'
        df = pd.read_csv(join(output_dir, res_dir, 'results.csv'))
        df['n_nodes'] = n_nodes
        df['n_time_steps'] = n_time_steps
        df[r'$\alpha$'] = f'{alpha}'
        data.append(df)


data = pd.concat(data)
color = 'lightgray'
fontsize = 16
titlesize = 16


fig, ax = plt.subplots(figsize=(15, 3), ncols=3, sharey=True)

sns.boxplot(x=r'$\alpha$', y='theta_rmse', color=color, showfliers=False, data=data, ax=ax[0])
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[0].set_title('Log-Odds [$\Theta(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r"$\alpha$", y='intercept_rmse', color=color, showfliers=False,data=data, ax=ax[1])
ax[1].set_xlabel(r"$\alpha$", fontsize=fontsize)
ax[1].set_title(r'Intercept [$\beta_1(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r'$\alpha$', y='coefs_rmse', color=color, showfliers=False, data=data, ax=ax[2])
ax[2].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[2].set_ylabel('', fontsize=fontsize)
ax[2].set_title(r'Coefficients [$\beta_2(t), \beta_3(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
fig.savefig('odds_coefs_alpha_sensitivity.pdf', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 3), ncols=3, sharey=True)

sns.boxplot(x=r'$\alpha$', y='U_rmse', color=color, showfliers=False, data=data, ax=ax[0])
ax[0].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_title('Latent Positions [$U_{1:2}(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r'$\alpha$', y='U_rmse_select', color=color, showfliers=False, data=data, ax=ax[1])
ax[1].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[1].set_ylabel('', fontsize=fontsize)
ax[1].set_title(r'Selected Latent Positions [$U_{1:\max(\hat{d}_0, 2)}(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r'$\alpha$', y='U_rmse_all', color=color, showfliers=False, data=data, ax=ax[2])
ax[2].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[2].set_ylabel('', fontsize=fontsize)
ax[2].set_title(r'All Latent Positions [$U(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)

fig.savefig('ls_alpha_sensitivity.pdf', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 3), ncols=3, sharey=True)

sns.boxplot(x=r'$\alpha$', y='proc_corr', color=color, showfliers=False, data=data, ax=ax[0])
ax[0].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[0].set_ylabel('Procrustes Correlation', fontsize=fontsize)
ax[0].set_title('Latent Positions [$U_{1:2}(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r'$\alpha$', y='proc_corr_select', color=color, showfliers=False, data=data, ax=ax[1])
ax[1].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[1].set_ylabel('', fontsize=fontsize)
ax[1].set_title('Selected Latent Positions [$U_{1:\max(\hat{d}_0, 2)}(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x=r'$\alpha$', y='proc_corr_all', color=color, showfliers=False, data=data, ax=ax[2])
ax[2].set_xlabel(r'$\alpha$', fontsize=fontsize)
ax[2].set_ylabel('', fontsize=fontsize)
ax[2].set_title(r'All Latent Positions [$U(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)


fig.savefig('ls_proc_corr_alpha_sensitivity.pdf', dpi=300, bbox_inches='tight')
plt.show()
