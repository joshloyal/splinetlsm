import pandas as pd
import numpy as np
import glob 

from os.path import join


res_dir_name = 'output'

for file_name in glob.glob(res_dir_name + '/*'):
    data = []
    for idx, res_file_name in enumerate(glob.glob(file_name + '/result_*csv')):
        df = pd.read_csv(res_file_name)
        df['sim_number'] = idx
        data.append(df)
    data = pd.concat(data)
    data.to_csv(join(file_name, 'results.csv'), index=False)

