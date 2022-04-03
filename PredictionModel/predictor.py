from PredictionModel.generator import generator
from sklearn.preprocessing import MinMaxScaler
from PredictionModel.read_data import read_data
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.signal import savgol_filter as sg
from time import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from PredictionModel.prediction_model import *
import matplotlib.pyplot as plt
np.random.seed(1333)


# def predictor(path, filename):#
def predictor(path, w, q, batch_size, units): #调用PSO算法时用
    print(path)
    read_col = [0, 1, 2, 3]
    temp_data = read_data(path, read_col)

    # 归一化
    temp_data = temp_data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_data = scaler.fit_transform(temp_data)
    data_or = temp_data

    # S-G平滑处理
    # temp_data = temp_data.reshape(1, -1)
    # temp_data = temp_data[0]
    # temp_data = sg(temp_data, w, q)
    # # temp_data = sg(temp_data, 15, 6)
    # temp_data = temp_data.reshape(-1, 1)

    look_back = 1  # 每1.5分钟预测一次
    step = 1  # 每10s采样一次
    delay = 1  # 预测10s
    batch_size = batch_size
    units = units
    # batch_size = 32
    # units = 128

    # hour * (60min/slot) * day
    train_max = 24 * 12 * 2
    val_max = 24 * 12 * 3

    # 生成构造器
    train_data = generator(temp_data, lookback=look_back, delay=delay, min_index=0, max_index=train_max,
                           batch_size=batch_size, step=step)
    val_data = generator(temp_data, lookback=look_back, delay=delay, min_index=train_max + 1, max_index=val_max,
                         batch_size=batch_size, step=step)
    test_data = generator(temp_data, lookback=look_back, delay=delay, min_index=val_max + 1, max_index=None,
                          batch_size=batch_size, step=step)

    val_steps = (val_max - train_max - 1 - look_back) // batch_size
    test_steps = (len(temp_data) - val_max - 1 - look_back) // batch_size
    # 创建模型
    callback = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_delta=0.00001,
                          mode='auto', cooldown=0, min_lr=0, verbose=0),
    ]
    start_t = time()
    model, model_name = build_model_lstm(input_shape=(look_back // step, 1), units=units)
    # model, model_name = build_RNN(input_shape=(look_back // step, 1), units=units)
    # model, model_name = build_GRU(input_shape=(look_back // step, 1), units=units)
    # model, model_name = build_CNN_BiLstm(input_shape=(look_back // step, 1), units=units)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    history = model.fit_generator(train_data, steps_per_epoch=int(len(temp_data) // batch_size), epochs=100, verbose=0,
                                  validation_data=val_data, validation_steps=val_steps, callbacks=callback,
                                  shuffle=False)

    end_t = time()
    print('Training use time', end_t - start_t)

    # 预测
    start_p = time()
    testPredict = model.predict(test_data, steps=test_steps)
    end_p = time()
    print('Predicting Use time:', end_p - start_p)


    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))

    data_11 = scaler.inverse_transform(data_or)

    raw_res = data_11[val_max + 1 + look_back:val_max + 1 + len(testPredict) + look_back, :]
    predict_res = testPredict

    # 计算测试机的mse
    # or_data = data_or[val_max + 1 + look_back:val_max + 1 + len(testPredict) + look_back, :]

    pred_mse = mean_squared_error(raw_res, testPredict)
    print('Predict mas is :', pred_mse)

    # 计算R²
    fit_r = r2_score(raw_res, testPredict)
    print('Fitting ability is ', fit_r)

    # 计算MAE
    pred_mae = mean_absolute_error(raw_res, testPredict)
    print("Predict mae is :", pred_mae)

    ploytestprediction = np.empty_like(temp_data)
    ploytestprediction[:, :] = np.nan
    ploytestprediction[val_max + 1 + look_back:val_max + 1 + len(testPredict) + look_back, :] = testPredict
    plt.plot(data_11, c='black', label='Actual cpu utilization')
    plt.plot(ploytestprediction, c='r', label='Predicted cpu utilization')
    plt.xlabel("Time stamp (5 mins)")
    plt.ylabel("CPU utilization")
    plt.legend()

    z = 0
    for i in range(len(ploytestprediction)):
        if np.isnan(ploytestprediction[i]):
            continue
        else:
            z = i
            break

    plt.vlines(z, 30, 50, color='b', linestyles='--')
    plt.savefig("../res/prediction_res111.pdf")
    plt.show()

    return pred_mse, fit_r, raw_res, predict_res


pred_mse, fit_r, raw_res, predict_res = predictor('../data/data_container__c_69452.csv', 21, 12, 43, 54)
