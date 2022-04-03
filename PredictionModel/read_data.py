import pandas as pd
from scipy.signal import savgol_filter as sg
import numpy as np

def read_data(path, read_col=[2]):
    data = pd.read_csv(path, usecols=read_col, header=None, engine='python')
    df = pd.DataFrame(data.values, columns=['machine_id', 'start_time', 'cpu', 'mem'])
    temp_data = df['mem'].values
    temp_data = temp_data.astype('float32')
    time_data = df['start_time'].values
    # print(temp_data.shape)

    # return temp_data, time_data
    return temp_data