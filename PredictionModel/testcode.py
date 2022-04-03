# import pandas as pd
#
# data = pd.read_csv('E:container/data_container__c_3.csv', header=None, engine='python')
# df = pd.DataFrame(data.values, columns=['container_id', 'machine_id', 'start_time', 'cpu', 'mem', 'disk'])
#
# temp = sum(df["cpu"].values)/(len(df["cpu"].values))
# print(temp)

from sko.PSO import PSO
import matplotlib.pyplot as plt
from PredictionModel.predictor import predictor
import numpy as np


def demo(x):
    # print("开始", x)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    x1 = x1.astype(int)
    x2 = x2.astype(int)
    x3 = x3.astype(int)
    x4 = x4.astype(int)
    # y1 = 999999
    if x1 > x2 and x1 % 2 == 1:
        print(x1, x2, x3, x4)
        y1, y2 = predictor("E:container_5min/data_container__c_34418.csv", x1, x2, x3, x4)
        if y2 < 0.5:
            return np.inf
        return y1
    else:
        print("重新迭代")
        return np.inf


constraint_ueq = (
    lambda x: 1 - (x[0] % 2),
    lambda x: x[1] - x[0] + 1,
)
# n_dim 代表参数的个数
# pop 粒子数 max_iter 最大迭代数 c1 c2 代表加速常数c1和c2是调整自身经验和社会经验在其运动中所起作用的权重
# w 惯性因子w对于粒子群算法的收敛性有较大的影响。w可以取[0,1]区间的随机数


max_iter = 10
pso = PSO(func=demo, n_dim=4, pop=20, max_iter=max_iter, w=0.65, c1=0.5, c2=0.5,
          lb=[1, 0, 8, 8], ub=[31, 30, 128, 256], constraint_ueq=constraint_ueq, verbose=True)
pso.run(max_iter=max_iter)
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

#
plt.plot(pso.gbest_y_hist)
plt.xlabel("Number of iteration")
plt.ylabel("Fitness value")
plt.savefig("res_fig/PSO_iter1.pdf")
plt.show()
