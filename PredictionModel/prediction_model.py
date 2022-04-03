from keras import Input, Model, layers
from keras import backend as k
import numpy as np
np.random.seed(1333)


def attention_block(inputs, flag=False):
    input_dim = int(inputs.shape[2])
    time_step = int(inputs.shape[1])
    a = layers.Permute((2, 1))(inputs)  # 置换输入的维度 为了将数据输入格式符合Dense的输入标准，索引从1开始
    a = layers.Reshape((input_dim, time_step))(a)  # 将输出调整为特征的形状， 做这一行没有用
    a = layers.Dense(time_step, activation='sigmoid')(a)
    if flag:
        a = layers.Lambda(lambda x: k.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)  # 置换过来，便于与LSTM衔接
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def build_CNN_BiLstm(input_shape, units):

    k.clear_session()
    pool = input_shape[0]
    input_shape = Input(shape=input_shape)
    CNN_1 = layers.Conv1D(filters=32, kernel_size=1, activation='relu')(input_shape)
    CNN_1 = layers.MaxPooling1D(pool_size=pool)(CNN_1)
    CNN_1 = layers.Dropout(0.3)(CNN_1)
    BiLstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(CNN_1)
    BiLstm = layers.add([input_shape, BiLstm])
    attention_out = attention_block(BiLstm)
    attention_out = layers.Flatten()(attention_out)
    output_data = layers.Dense(1, activation='sigmoid')(attention_out)
    model = Model(input_shape, output_data)
    model.summary()
    name = 'CNN_BiLSTM'
    return model, name


def build_model_lstm(input_shape, units):
    k.clear_session()
    input_data = Input(shape=input_shape)
    BiLstm_1 = layers.LSTM(units, return_sequences=False)(input_data)
    # BiLstm_1 = attention_block(BiLstm_1)
    # BiLstm_1 = layers.Dropout(0.4)(BiLstm_1)
    output_data = layers.Dense(1, activation='sigmoid')(BiLstm_1)
    model = Model(input_data, output_data)
    model.summary()
    name = 'LSTM'
    return model, name


def build_GRU(input_shape, units):

    k.clear_session()
    input_data = Input(shape=input_shape)
    gru = layers.GRU(units, return_sequences=False)(input_data)
    output_data = layers.Dense(1, activation="sigmoid")(gru)
    model = Model(input_data, output_data)
    model.summary()
    name = 'GRU'
    return model, name


def build_RNN(input_shape, units):
    k.clear_session()
    input_shape = Input(shape=input_shape)
    rnn = layers.SimpleRNN(units=units, return_sequences=False)(input_shape)
    output_data = layers.Dense(1, activation="sigmoid")(rnn)
    model = Model(input_shape, output_data)
    model.summary()
    name = 'RNN'
    return model, name


def build_CNN_BiLstm(input_shape, units):

    k.clear_session()
    pool = input_shape[0]
    input_shape = Input(shape=input_shape)
    CNN_1 = layers.Conv1D(filters=64, kernel_size=1, activation='relu')(input_shape)
    CNN_1 = layers.MaxPooling1D(pool_size=pool)(CNN_1)
    CNN_1 = layers.Dropout(0.3)(CNN_1)
    BiLstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(CNN_1)
    BiLstm = layers.add([input_shape, BiLstm])
    attention_out = attention_block(BiLstm)
    attention_out = layers.Flatten()(attention_out)
    output_data = layers.Dense(1, activation='sigmoid')(attention_out)
    model = Model(input_shape, output_data)
    # model.summary()
    name = 'CNN_BiLSTM'
    return model, name