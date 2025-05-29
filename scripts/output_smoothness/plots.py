import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


color = 'lightgray'
fontsize = 12
titlesize = 14
label_map = {
        0.5 : r'Matern($\nu=0.5$)',
        1.5 : r'Matern($\nu=1.5$)',
        2.5 : r'Matern($\nu=2.5$)',
        'gp': r'Exponential'
}
data = []
for k, nu in enumerate([2.5, 1.5, 0.5]):
    df = pd.read_csv(f'output/matern_nu{nu}_n200_T100_d0.2/results.csv')
    df['nu'] = label_map[nu]
    data.append(df)
df = pd.read_csv(f'output/gp_nu0.5_n200_T100_d0.2/results.csv')
df['nu'] = label_map['gp']
data.append(df)
data = pd.concat(data)

fig, ax = plt.subplots(figsize=(16, 5), nrows=2, ncols=2, sharey=True)
ax = ax.ravel()

sns.boxplot(x='nu', y='theta_rmse', order=label_map.values(), data=data, ax=ax[0], showfliers=False, color=color)
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel('', fontsize=fontsize)
ax[0].set_title('Log-Odds [$\Theta(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='y', which='major', labelsize=fontsize)
ax[0].get_xaxis().set_ticklabels([])


sns.boxplot(x='nu', y='coefs_rmse', order=label_map.values(), data=data, ax=ax[2], showfliers=False, color=color)
ax[2].set_xlabel('GP Covariance', fontsize=fontsize)
ax[2].set_title(r'Coefficients [$\beta_2(t), \beta_3(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='y', which='major', labelsize=fontsize)
ax[2].set_ylabel('RMSE', fontsize=fontsize)
ax[2].tick_params(axis='x', which='major', labelsize=fontsize)

sns.boxplot(x='nu', y='intercept_rmse', order=label_map.values(), data=data, ax=ax[3], showfliers=False, color=color)
ax[3].set_xlabel('GP Covariance', fontsize=fontsize)
ax[3].set_title(r'Intercept [$\beta_1(t)$]', fontsize=titlesize)
ax[3].tick_params(axis='y', which='major', labelsize=fontsize)
ax[3].tick_params(axis='x', which='major', labelsize=fontsize)

sns.boxplot(x='nu', y='U_rmse_select', order=label_map.values(), data=data, ax=ax[1], showfliers=False, color=color)
ax[1].set_xlabel('', fontsize=fontsize)
ax[1].set_title('Latent Positions [$U(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='y', which='major', labelsize=fontsize)
ax[1].get_xaxis().set_ticklabels([])

fig.savefig('recovery_smoothness.pdf', dpi=300, bbox_inches='tight')

plt.show()

fig, ax = plt.subplots(figsize=(25, 4), ncols=3, sharey=True)

sns.boxplot(x='nu', y='U_rmse', order=label_map.values(), data=data, ax=ax[0], showfliers=False, color=color)
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel('GP Covariance', fontsize=fontsize)
ax[0].set_title('Latent Positions [$U_{1:2}(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='y', which='major', labelsize=fontsize)
ax[0].tick_params(axis='x', which='major', labelsize=12)


sns.boxplot(x='nu', y='U_rmse_select', order=label_map.values(), data=data, ax=ax[1], showfliers=False, color=color)
ax[1].set_xlabel('GP Covariance', fontsize=fontsize)
ax[1].set_title('Selected Latent Positions [$U_{1:\hat{d}}(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='y', which='major', labelsize=fontsize)
ax[1].tick_params(axis='x', which='major', labelsize=12)

sns.boxplot(x='nu', y='U_rmse_all', order=label_map.values(), data=data, ax=ax[2], showfliers=False, color=color)
ax[2].set_xlabel('GP Covariance', fontsize=fontsize)
ax[2].set_title('All Latent Positions [$U(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='y', which='major', labelsize=fontsize)
ax[2].tick_params(axis='x', which='major', labelsize=12)


fig.savefig('ls_recovery_smoothness.pdf', dpi=300, bbox_inches='tight')

plt.show()

fig, ax = plt.subplots(figsize=(25, 4), ncols=3, sharey=True)

sns.boxplot(x='nu', y='proc_corr', order=label_map.values(), data=data, ax=ax[0], showfliers=False, color=color)
ax[0].set_ylabel('Procrustes Correlation', fontsize=fontsize)
ax[0].set_xlabel('GP Covariance', fontsize=fontsize)
ax[0].set_title('Latent Positions [$U_{1:2}(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='y', which='major', labelsize=fontsize)
ax[0].tick_params(axis='x', which='major', labelsize=12)


sns.boxplot(x='nu', y='proc_corr_select', order=label_map.values(), data=data, ax=ax[1], showfliers=False, color=color)
ax[1].set_xlabel('GP Covariance', fontsize=fontsize)
ax[1].set_title('Selected Latent Positions [$U_{1:\hat{d}}(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='y', which='major', labelsize=fontsize)
ax[1].tick_params(axis='x', which='major', labelsize=12)

sns.boxplot(x='nu', y='proc_corr_all', order=label_map.values(), data=data, ax=ax[2], showfliers=False, color=color)
ax[2].set_xlabel('GP Covariance', fontsize=fontsize)
ax[2].set_title('All Latent Positions [$U(t)$]', fontsize=titlesize)
ax[2].tick_params(axis='y', which='major', labelsize=fontsize)
ax[2].tick_params(axis='x', which='major', labelsize=12)


fig.savefig('ls_proc_corr_recovery_smoothness.pdf', dpi=300, bbox_inches='tight')

plt.show()
