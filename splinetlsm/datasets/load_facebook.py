import joblib
import numpy as np
import pandas as pd

from datetime import datetime as dt
from os.path import dirname, join


__all__ = ['load_facebook']


def load_facebook():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'facebook', 'facebook_filtered.gz')
    Y = joblib.load(open(file_name, 'rb'))
   
    time_points = np.arange(len(Y))
    time_labels = []
    years = [2007, 2008, 2009]
    for t in np.arange(len(Y)+1):
        month = t % 12 + 1
        year = years[t // 12] 
        #date = dt.fromisocalendar(year, week, 1)
        #time_labels.append(date.strftime("%Y-%m-%W"))
        time_labels.append(f"{month}-{year}")


    return Y, time_points, time_labels
