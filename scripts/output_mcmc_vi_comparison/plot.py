import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('sim_n100_T10/results.csv')

fig, ax = plt.subplots(ncols=2, figsize=(20,6), sharex=True, sharey=True)
x_min = data['coef1_mcmc'].min()
ax[0].scatter(data['coef1_mcmc'], data['coef1_svi'], c='darkgray', edgecolor='k')
ax[0].axline((x_min, x_min), slope=1, linestyle='--', lw=2, c='k')
ax[0].set_xlabel('Width of 95% CI (MCMC)', fontsize=16)
ax[0].set_ylabel('Width of 95% CI (SVI)', fontsize=16)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].set_title(r'$\beta_1(t)$', fontsize=18)

x_min = data['coef2_mcmc'].min()
ax[1].scatter(data['coef2_mcmc'], data['coef2_svi'], c='darkgray', edgecolor='k')
ax[1].axline((x_min, x_min), slope=1, linestyle='--', lw=2, c='k')
ax[1].set_xlabel('Width of 95% CI (MCMC)', fontsize=16)
ax[1].set_ylabel('', fontsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=12, labelleft=True)
ax[1].set_title(r'$\beta_2(t)$', fontsize=18)

fig.savefig('coef_mcmc_svi_comp.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(ncols=3, figsize=(20,6))

stat_label = ['Density', 'Transitivity', 'Degree of Node 1']
for k, stat in enumerate(['density', 'transitivity', 'degree']):
    x_min = data[f'{stat}_mcmc'].min()
    ax[k].scatter(data[f'{stat}_mcmc'], data[f'{stat}_svi'], c='darkgray', edgecolor='k')
    ax[k].axline((x_min, x_min), slope=1, linestyle='--', lw=2, c='k')
    ax[k].set_xlabel('Width of 95% CI (MCMC)', fontsize=16)
    if k == 0:
        ax[k].set_ylabel('Width of 95% CI (SVI)', fontsize=16)
    ax[k].tick_params(axis='both', which='major', labelsize=12, labelleft=True)
    ax[k].set_title(stat_label[k], fontsize=18)


fig.savefig('posterior_predictive_mcmc_svi_comp.pdf', dpi=300, bbox_inches='tight')
plt.show()
