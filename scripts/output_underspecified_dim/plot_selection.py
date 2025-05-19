import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join


output_dir = 'output'
node_sizes = [100, 250, 500, 1000]
n_time_steps = 100
densities = [0.1, 0.2, 0.3]

data = []
for density in densities:
    res_dir = f'gp_n200_T100_d{density}'
    df = pd.read_csv(join(output_dir, res_dir, 'results.csv'))
    df = df.loc[pd.isna(df['zero_mse_perc']), :]
    df['Expected Density'] = str(density)
    df['selected'] = df['n_features'] == 2
    data.append(df)


data = pd.concat(data)
fontsize = 12
titlesize = 14

agg = data.groupby(['Expected Density'])[['n_features', 'selected']].agg(['mean', 'median', 'std'])
print(agg)

