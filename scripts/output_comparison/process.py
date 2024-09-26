import pandas as pd
import numpy as np
import glob 

from os.path import join


for file_name in glob.glob('lady_n*'):
    data = []
    for res_file_name in glob.glob(file_name + '/result_*csv'):
        data.append(pd.read_csv(res_file_name))
    data = pd.concat(data)
    data.to_csv(join(file_name, 'results.csv'), index=False)


for file_name in glob.glob('fase_n*'):
    data = []
    for res_file_name in glob.glob(file_name + '/result_*csv'):
        data.append(pd.read_csv(res_file_name))
    data = pd.concat(data)
    data.to_csv(join(file_name, 'results.csv'), index=False)


for file_name in glob.glob('sim_n*'):
    data = []
    for res_file_name in glob.glob(file_name + '/result_*csv'):
        data.append(pd.read_csv(res_file_name))
    data = pd.concat(data)
    data.to_csv(join(file_name, 'results.csv'), index=False)
