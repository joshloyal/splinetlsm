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
        df['selected'] = df['n_features'] == 2
        data.append(df)
data = pd.concat(data)

agg = data.groupby(['n_time_steps', 'Expected Density'])[['n_features', 'selected']].agg(['mean', 'median', 'std'])
print(agg)
