import os
import pandas as pd


# 下列三个函数是将文件夹中的所有文件操作， add_row.py是单个文件的操作
# 将文件按时间戳排序
def sort_csv(file_list, file_dir, save_dir):
    for files in file_list:
        files_path = file_dir + '/' + files
        save_path = save_dir + '/' + files
        print(files_path)

        # read data from path
        data = pd.read_csv(files_path, header=None, engine='python')
        df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])

        # create dataframe same as data
        df_empty = pd.DataFrame(columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
        # sort
        df_empty = df.sort_values(by=["start_time"])
        # save file after sort to save_path
        df_empty.to_csv(save_path, index=False, header=False, mode='a')


# 利用全部cpu的均值将数据集补全（使数据集每隔10s记录一次）
def add_row(file_list, file_dir, save_dir):
    for files in file_list:
        files_path = file_dir + '/' + files
        save_path = save_dir + '/' + files

        data = pd.read_csv(files_path, header=None, engine='python')
        df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
        df_empty = pd.DataFrame(columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
        time = df['start_time'].values
        df_cpu = df['cpu'].values
        x = sum(df_cpu) / len(df_cpu)
        # 容器id列表
        c_id = df['container_id'].values
        # 机器id列表
        m_id = df['machine_id'].values
        df_cid = c_id[0]
        df_mid = m_id[0]
        print(type(time), time)
        print(df.index)
        sum1 = 0
        for i in range(0, len(time) - 1):
            rows = int((time[i + 1] - time[i]) / 10)
            if rows != 1 and rows <= 10:
                for j in range(0, rows):
                    df_empty = df_empty.append([{'container_id': df_cid,
                                                 'machine_id': df_mid,
                                                 'start_time': time[i] + (j + 1) * 10,
                                                 'cpu': x}], ignore_index=True)
                    sum1 = sum1 + 1
        print(sum1)
        print(df_empty.values)
        print(max(df['cpu'].values))
        df_empty.to_csv(save_path, index=False, header=False, mode='a')


# 合并数据集（每隔5min）
def sum_row(file_list, file_dir, save_dir):
    for files in file_list:
        files_path = file_dir + '/' + files
        save_path = save_dir + '/' + files

        data = pd.read_csv(files_path, header=None, engine='python')
        df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
        df_empty = pd.DataFrame(columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
        df_cpu = df['cpu'].values
        df_time = df['start_time'].values
        # 容器id列表
        c_id = df['container_id'].values
        # 机器id列表
        m_id = df['machine_id'].values
        df_cid = c_id[0]
        df_mid = m_id[0]
        t = 30  # 控制合并时间的大小
        for i in range(0, len(df_cpu), t):
            temp = int(sum(df_cpu[i:i+t])/len(df_cpu[i:t+i]))
            df_empty = df_empty.append([{'container_id': df_cid,
                                        'machine_id': df_mid,
                                         'cpu': temp}], ignore_index=True)
        df_empty.to_csv(save_path, index=False, header=False, mode='a')


file_dir = "E:container_5min_2"
save_dir = "E:container_5min"
file_list = os.listdir(file_dir)
print(len(file_list), file_list)

# add_row(file_list, file_dir, save_dir)
# sort_csv(file_list, file_dir, save_dir)
sum_row(file_list, file_dir, save_dir)