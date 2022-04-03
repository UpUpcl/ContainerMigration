import os
from utils.preprocessing_file import sum_row
import pandas as pd

filedir = "E:container_5min_2"
filelist = os.listdir(filedir)
savedir = "E:container_15min"


# # 操作合并
# for filename in filelist:
#     path = filedir + '/' + filename
#     save_path = savedir + '/' + filename
#     data = pd.read_csv(path, header=None, engine='python')
#     df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
#     sum_row(df, save_path)



filedir_1 = "E:container_5min"
filelist_1 = os.listdir(filedir_1)
path_1 = []

filedir_2 = "E:prediction_res_15min"
filelist_2 = os.listdir(filedir_2)
path_2 = []

for index in filelist_1:
    path_1.append(index)

for index in filelist_2:
    path_2.append(index)

for file in filelist_1:
    if file not in path_2:
        print(file)
