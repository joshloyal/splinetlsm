import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


output_dir = 'output'
node_sizes = [100, 200, 300]
n_time_steps = 100
density = 0.2 

fontsize = 12
titlesize = 14
fig, ax = plt.subplots(figsize=(20, 4), nrows=4, ncols=4, sharey='row')
for m, metric in enumerate(['theta_rmse', 'coefs_rmse', 'intercept_rmse', 'U_rmse']):
    data = pd.read_csv(f'output/gp_nu0.5_n200_T100_d0.2/results.csv')
    data.loc[:, 'color'] = 'normal'
    data.loc[data['k'] == 2, 'color'] = 'default'
    data_best = data.query('is_best')
    data_best.loc[:, 'spline_dimension'] = 'WAIC' 
    data_best.loc[:, 'color'] = 'WAIC'
    data = pd.concat((data, data_best), ignore_index=True)
    sns.boxplot(x='spline_dimension', y=metric, hue='color', data=data, ax=ax[m, 0], showfliers=False, legend=False)
    if m == 0:
        print('Feature Selection Percent (Exponential): {}'.format(np.mean(data['n_features'] == 4)))
        ax[m, 0].set_title('Exponential')

    for k, nu in enumerate([2.5, 1.5, 0.5]):
        data = pd.read_csv(f'output/matern_nu{nu}_n200_T100_d0.2/results.csv')
        data.loc[:, 'color'] = 'normal'
        data.loc[data['k'] == 2, 'color'] = 'default'

        data_best = data.query('is_best')
        data_best.loc[:, 'spline_dimension'] = 'WAIC' 
        data_best.loc[:, 'color'] = 'WAIC'

        data = pd.concat((data, data_best), ignore_index=True)
        sns.boxplot(x='spline_dimension', y=metric, hue='color', data=data, ax=ax[m, k+1], showfliers=False, legend=False)
        if m == 0:
            print('Feature Selection Percent ({}): {}'.format(nu, np.mean(data['n_features'] == 4)))
            ax[m, k+1].set_title(r'Matern($\nu = $' + f'{nu})')

plt.show()
