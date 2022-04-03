import os
import pandas as pd
import numpy as np


def read_data(path, read_col=[0, 1]):
    data = pd.read_csv(path, usecols=read_col, header=None, engine='python')
    df = pd.DataFrame(data.values, columns=['real_value', 'prediction_value'])
    real_value = df['real_value'].values
    prediction_value = df['prediction_value'].values
    return real_value, prediction_value


def save_res(path, data):
    df_empty = pd.DataFrame(columns=['data'])
    for i in range(len(data)):
        df_empty = df_empty.append([{'data': data[i][0]}], ignore_index=True)
    df_empty.to_csv(path, index=False, header=False, mode='a')


if __name__ == '__main__':

    filedir = "E:prediction_res_15min"
    filelist = os.listdir(filedir)
    num = -1
    real_value = np.zeros([80, 1])
    prediction_value = np.zeros([80, 1])

    for filename in filelist[:80]:
        path = filedir + '/' + filename
        num += 1
        print(path, num)
        real, prediction = read_data(path)
        real_value[num] = real[0]
        prediction_value[num] = prediction[0]

    save_res("E:prediction_cpu_15min.csv", prediction_value)
    save_res("E:real_cpu_15min.csv", real_value)
    # print(real_value, '\n\n\n', prediction_value)