import pandas as pd


def read_preediction_value(path,containerNum):
    data = pd.read_csv(path, usecols=[0], header=None, engine='python')
    df = pd.DataFrame(data.values, columns=['prediction_value'])
    if containerNum > len(df['prediction_value']):
        print("输入的容器数量大于已有的容器数量，因此默认全部取出")
        containerNum = len(df['prediction_value'])
    prediction_value = df['prediction_value'][:containerNum].values
    return prediction_value


def read_real_value(path, containerNum):
    data = pd.read_csv(path, usecols=[0], header=None, engine='python')
    df = pd.DataFrame(data.values, columns=['real_value'])
    if containerNum > len(df['real_value']):
        print("输入的容器数量大于已有的容器数量，因此默认全部取出")
        containerNum = len(df['real_value'])
    real_value = df['real_value'][:containerNum].values
    return real_value
