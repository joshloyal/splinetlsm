import pandas as pd
import numpy as np
import glob 

from os.path import join


res_dir_name = 'sim_n100_T10'

for file_name in glob.glob(res_dir_name + '/*'):
    data = []
    for res_file_name in glob.glob(file_name + '/result_*csv'):
        data.append(pd.read_csv(res_file_name))
    data = pd.concat(data)
    data.to_csv(join(file_name, 'results.csv'), index=False)

