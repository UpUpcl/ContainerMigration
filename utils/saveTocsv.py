import pandas as pd


def saveToexcel(filename, mse, r, w, p, b, u):
    df_empty = pd.DataFrame(columns=['filename', 'mse', 'r', 'w', 'p', 'u', 'b'])
    df_empty = df_empty.append([{'filename': filename,
                                 'mse': mse,
                                 'r': r,
                                 'w': w,
                                 'p': p,
                                 'u': u,
                                 'b': b}], ignore_index=True)
    df_empty.to_csv("E:prediction_res_15min/results.csv", index=False, header=False, mode='a')


def save_res(path, data, prediction):
    df_empty = pd.DataFrame(columns=['data', 'PredictionModel'])
    for i in range(len(data)):
        df_empty = df_empty.append([{'data': data[i][0],
                                 'PredictionModel': prediction[i][0]}], ignore_index=True)
    df_empty.to_csv(path, index=False, header=False, mode='a')