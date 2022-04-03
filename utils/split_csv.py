import pandas as pd
import numpy as np
from numpy import NaN


chunk_iterator = pd.read_csv("G:/alibaba_clusterdata_v2018/container_usage.csv",
                             chunksize=50000, iterator=True, usecols=[0, 1, 2, 3, 6, 8],
                             header=None)
i = 1

# 找到列为某个值的
# def chunk_manipulate(chunk):
#     df = pd.DataFrame(chunk.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
#     data = df[df['container_id'].isin(['c_3'])]
#     path = 'F:/container_10.csv'
#     data.to_csv(path, index=False, header=False, mode='a')

# 将containers_usage大型文件，按照container_id拆分
def chunk_manipulate(chunk):
    # data = chunk.values
    df = pd.DataFrame(chunk.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
    # df = df[df.machine_id == 'm_1']
    # isin()传入一个要匹配的name_list
    c_id = df['container_id'].values
    cc_id = []
    for i in c_id:
        if i not in cc_id:
            cc_id.append(i)
    print("111", len(cc_id), cc_id[:10])
    for j in cc_id:
        if j == NaN:
            j = 'c_0'
        # path = 'F:/test/data_container_%s.csv' % j
        data = df[df['container_id'].isin([j])]
        print("2222", len(data))

        path = "E:/container/data_container__%s.csv" % j
        print(path)
    # df = df.dropna(subset=['cpu', 'mem'])
        data.to_csv(path, index=False, header=False, mode="a")
    # print(df.values)
    # print(type(df))
    # return df


for chunk in chunk_iterator:

    chunk_manipulate(chunk)

    print("hhhhhhhhhhhhhhhhhhhh", i)
    i = i + 1


