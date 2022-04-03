from PredictionModel.generator import generator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PredictionModel.read_data import read_data
import tensorflow as tf
from scipy.signal import savgol_filter as sg

from keras.models import load_model


def testPredict(testPredict):
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
    testPredictPlot = np.empty_like(temp_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[24 * 12 * 3 + 1 + look_back:24 * 12 * 3+1 + len(testPredict) + look_back, :] = testPredict
    # testPredictPlot[0 + look_back:0 + len(testPredict) + look_back, :] = testPredict
    return testPredictPlot


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    np.random.seed(1333)
    # read data
    path = "E:container_5min/data_container__c_34418.csv"
    read_col = [0, 1, 2, 3]
    temp_data = read_data(path, read_col)
    # print('111111111111111', time_data.shape, type(time_data))
    # time_data = time_data.reshape(-1, 1)
    # 归一化
    temp_data = temp_data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_data = scaler.fit_transform(temp_data)

    # S-G平滑处理
    temp_data1 = temp_data.reshape(1, -1)
    temp_data1 = temp_data1[0]
    temp_data1 = sg(temp_data1, 9, 2)
    temp_data1 = temp_data1.reshape(-1, 1)

    look_back = 1  # 利用前3分钟
    step = 1  # 每1分钟采样一次
    delay = 1  # 预测10s
    batch_size = 32
    batch_size_8 = 8

    units_145 = 145
    units = 128
    path = '../save_model/'

    train_max = 24 * 12 * 2
    val_max = 24 * 12 * 3

    # 构造器

    test_data1 = generator(temp_data, lookback=look_back, delay=delay, min_index=val_max+1, max_index=None,
                          batch_size=batch_size, step=step)
    test_data2 = generator(temp_data, lookback=look_back, delay=delay, min_index=val_max+1, max_index=None,
                          batch_size=batch_size, step=step)
    test_data3 = generator(temp_data, lookback=look_back, delay=delay, min_index=val_max+1, max_index=None,
                          batch_size=batch_size, step=step)
    test_data4 = generator(temp_data, lookback=look_back, delay=delay, min_index=val_max+1, max_index=None,
                          batch_size=batch_size, step=step)

    test_data_pso = generator(temp_data1, lookback=look_back, delay=delay, min_index=val_max+1, max_index=None,
                          batch_size=batch_size_8, step=step)

    test_steps = (len(temp_data) - val_max - 1 - look_back) // batch_size
    test_steps_pso = (len(temp_data) - val_max - 1 - look_back) // batch_size_8

    # 加载模型
    PSL = load_model(path+'PSL.h5')
    SL = load_model(path+'SL.h5')
    LSTM = load_model(path+'LSTM.h5')
    RNN = load_model(path+'RNN.h5')
    GRU = load_model(path+'GRU.h5')

    # 预测
    PSL_prediction = PSL.predict(test_data_pso, steps=test_steps_pso)
    SL_prediction = SL.predict(test_data1, steps=test_steps)
    LSTM_prediction = LSTM.predict(test_data2, steps=test_steps)
    RNN_prediction = RNN.predict(test_data3, steps=test_steps)
    GRU_prediction = GRU.predict(test_data4, steps=test_steps)

    PSL_prediction = testPredict(PSL_prediction)
    SL_prediction = testPredict(SL_prediction)
    LSTM_prediction = testPredict(LSTM_prediction)
    RNN_prediction = testPredict(RNN_prediction)
    GRU_prediction = testPredict(GRU_prediction)

    indices = range(1, len(temp_data) - 1, 10)
    data_11 = scaler.inverse_transform(temp_data)
    # x_data = time_data[indices]  # 时间戳
    data_1 = data_11[indices]  # 原数据
    data_pso = PSL_prediction[indices]
    data_sl = SL_prediction[indices]
    data_lstm = LSTM_prediction[indices]
    data_rnn = RNN_prediction[indices]
    data_gru = GRU_prediction[indices]

    plt.figure(figsize=(9, 6))
    # plt.plot(x_data, data_1, c='black', label='data', lw=1)
    # plt.plot(x_data, data_pso, c='red', label='PSL', lw=1)
    # plt.plot(x_data, data_sl, c='coral', linestyle='-.', label='SL', lw=1)
    # plt.plot(x_data, data_lstm, c='fuchsia', linestyle='--', label='LSTM', lw=1)
    # plt.plot(x_data, data_rnn, c='c', linestyle='--', marker='*', ms='1.5', label='RNN', lw=1)
    # plt.plot(x_data, data_gru, c='y', linestyle=':', label='GRU', lw=1)
    plt.plot(data_1, c='black', label='data', lw=1)
    plt.plot(data_pso, c='red', label='PSL', lw=1)
    plt.plot(data_sl, c='c', linestyle='-.', label='SL', lw=1)
    # plt.plot(data_lstm, c='fuchsia', linestyle='--', label='LSTM', lw=1)
    # plt.plot(data_rnn, c='c', linestyle='--', marker='*', ms='1.5', label='RNN', lw=1)
    # plt.plot(data_gru, c='olivedrab', linestyle=':', label='GRU', lw=1)
    plt.tick_params(labelsize=15)
    plt.xlabel('Time stamp (5mins)', fontdict={'size': 15})
    plt.ylabel('CPU Usage(%)', fontdict={'size': 15})
    plt.legend(loc='upper left', borderpad=1.5, labelspacing=1.5, prop={"size": 15})

    # x = 0
    # for i in indices:
    #     if np.isnan(PSL_prediction[i]):
    #         continue
    #     else:
    #         x = i
    #         break
    #
    # print(x, time_data[x])
    # plt.vlines(time_data[x], 10, 90, color='b', linestyles='--')
    plt.savefig('../res_fig/performance1.pdf')
    plt.show()