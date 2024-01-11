import pandas as pd
import numpy as np
import glob 

from os.path import join

density = [0.1, 0.2, 0.3]
sims = [[100, 10], [100, 20], [200, 10]]

for (n, T) in sims:
    for d in density:
        print((n, T, d))
        file_name = f"n{n}_T{T}_d{d}/results.csv"

        data1 = pd.read_csv('fase_' +  file_name)
        data1 = pd.read_csv('sim_' +  file_name)
        data = pd.concat((data1, data2), axis=1)
        data = pd.concat((data.mean(axis=0), data.std(axis=0)), axis=1)
        print(data.iloc[[9, 2, 13, 11, 6, -1]])
        print()
