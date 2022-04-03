import tensorflow as tf
import numpy as np
import os
from PredictionModel.predictor import predictor
from sko.PSO import PSO
import matplotlib.pyplot as plt
from utils.saveTocsv import saveToexcel, save_res
from sklearn.metrics import r2_score


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
        global path
        y1, y2, c, _ = predictor(path, x1, x2, x3, x4)
        # if y2 < 0.1:
        #     return np.inf
        return y1
    else:
        print("重新迭代")
        return np.inf


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    np.random.seed(1333)
    # read data
    filedir = "E:test"
    filelist = os.listdir(filedir)
    num = 0
    # 将所有文件进行预测，并将mse和r^2存储
    constraint_ueq = (
        lambda x: 1 - (x[0] % 2),
        lambda x: x[1] - x[0] + 1,
    )
    for filename in filelist:
        global path
        path = filedir + '/' + filename
        print(path)
        path1 = "E:prediction_res_15min" + '/' + filename
        num = num + 1

        max_iter = 10
        pso = PSO(func=demo, n_dim=4, pop=20, max_iter=max_iter, w=0.65, c1=0.5, c2=0.5,
                  lb=[1, 0, 8, 8], ub=[31, 30, 64, 128], constraint_ueq=constraint_ueq, verbose=True)
        pso.run(max_iter=max_iter)
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

        plt.plot(pso.gbest_y_hist)
        plt.xlabel("Number of iteration")
        plt.ylabel("Fitness value")
        # # plt.savefig("res_fig/PSO_iter1.pdf")
        plt.show()
        print("PSO已经完成！！！！且最优参数为：", pso.gbest_x, pso.gbest_y)
        if pso.gbest_y == np.inf:
            print(path, "预测失败！！！！！！")
            continue
        else:
            pred_mse, fit_r, data, prediction_res = predictor(path, int(pso.gbest_x[0]), int(pso.gbest_x[1]), int(pso.gbest_x[2]), int(pso.gbest_x[3]))
            print(r2_score(data, prediction_res))
            saveToexcel(filename, pred_mse, fit_r, int(pso.gbest_x[0]), int(pso.gbest_x[1]), int(pso.gbest_x[2]), int(pso.gbest_x[3]))
            save_res(path1, data, prediction_res)
        print(num)