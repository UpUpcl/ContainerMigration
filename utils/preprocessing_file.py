import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 由于每个containers的使用情况数据存在丢失，因此利用将文件每隔10秒记录一次，其中cpu利用均值补全 （改函数的功能为补全数据）
def add_row(df):
    df_empty = pd.DataFrame(columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
    time = df['start_time'].values
    df_cpu = df['cpu'].values
    x = sum(df_cpu)/len(df_cpu)
    # 容器id列表
    c_id = df['container_id'].values
    # 机器id列表
    m_id = df['machine_id'].values
    df_cid = c_id[0]
    df_mid = m_id[0]
    print(type(time), time)
    print(df.index)
    sum = 0
    for i in range(0, len(time) - 1):
        rows = int((time[i + 1] - time[i]) / 10)
        if rows != 1 and rows <= 10:
            for j in range(0, rows):
                df_empty = df_empty.append([{'container_id': df_cid,
                                             'machine_id': df_mid,
                                             'start_time': time[i] + (j + 1) * 10,
                                             'cpu':x}], ignore_index=True)
                sum = sum + 1
    print(sum)
    print(df_empty.values)
    print(max(df['cpu'].values))
    df_empty.to_csv('F:/data_container_1_1.csv', index=False, header=False, mode='a')


# 由于每10s记录一次数据，此处是将使用情况按照每五分钟合并一次，并且利用该段时间的均值作为新的时间戳的cpu利用率 （合并数据集）
def sum_row(df, path):
    df_empty = pd.DataFrame(columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
    df_cpu = df['cpu'].values
    df_time = df['start_time'].values
    # 容器id列表
    c_id = df['container_id'].values
    # 机器id列表
    m_id = df['machine_id'].values
    df_cid = c_id[0]  # c_id中的所有id一样，因此去index为0的
    df_mid = m_id[0]  # 同上
    t = 90  # t控制合并区间的大小
    for i in range(0, len(df_cpu), t):
        temp = int(sum(df_cpu[i:i+t])/len(df_cpu[i:t+i]))  # 新时间戳的cpu利用率
        df_empty = df_empty.append([{'container_id': df_cid,
                                    'machine_id': df_mid,
                                     'cpu': temp}], ignore_index=True)
    df_empty.to_csv(path, index=False, header=False, mode='a')


# data = pd.read_csv('E:container/data_container__c_3.csv', header=None, engine='python')
# df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
# add_row(df)
# # sum_row(df)
# plt.plot()
