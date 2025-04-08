import plac
import os
import numpy as np

from splinetlsm.datasets import load_polecat

def make_network():
    Y, time_points, X, node_names, iso_codes, regions, time_labels = load_polecat()
    
    out_dir = f'fase_data'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    np.savetxt(os.path.join(out_dir, 'time_points.npy'), time_points)
    for t in range(Y.shape[0]):
        np.savetxt(os.path.join(out_dir, f'Y_{t+1}.npy'), Y[t])


if __name__ == '__main__':
    plac.call(make_network)
