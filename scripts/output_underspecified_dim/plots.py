import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


output_dir = 'output'
densities = [0.1, 0.2, 0.3]
values = np.array(['d = 6', 'd = 2'])

data = []
for density in densities:
    res_dir = f'gp_n200_T100_d{density}'
    df = pd.read_csv(join(output_dir, res_dir, 'results.csv'))
    df['Expected Density'] = str(density)
    df['Latent Dimension'] = values[pd.isna(df['zero_mse_perc']).astype(int)]

    data.append(df)


data = pd.concat(data)
fontsize = 14
titlesize = 16

fig, ax = plt.subplots(figsize=(15, 3), ncols=3, sharey=True)

sns.boxplot(x='Expected Density', y='theta_rmse', hue='Latent Dimension', data = data, ax=ax[0], fliersize=0)
ax[0].set_ylabel('RMSE', fontsize=fontsize)
ax[0].set_xlabel('Expected Density', fontsize=fontsize)
ax[0].set_title('Log-Odds [$\Theta(t)$]', fontsize=titlesize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x='Expected Density', y='total_coefs_rmse', hue='Latent Dimension', data = data, ax=ax[1], fliersize=0)
ax[1].set_xlabel('Expected Density', fontsize=fontsize)
ax[1].set_title(r'Coefficients [$\beta(t)$]', fontsize=titlesize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

sns.boxplot(x='Expected Density', y='UUt_rmse', hue='Latent Dimension', data = data, ax=ax[2], fliersize=0)
ax[2].set_xlabel('Expected Density', fontsize=fontsize)
ax[2].set_title(r'Latent Similarity [$U(t)U(t)^{\top}$]', fontsize=titlesize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)

fig.savefig('underspecified_recovery.pdf', dpi=300, bbox_inches='tight')
