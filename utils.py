import json # package to read json code
import requests # package to get data from an API
import pandas as pd # package for data frames
import datetime # package to deal with time
import matplotlib.pyplot as plt #package to do plotting
from scipy import signal
from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def preprocess(df,type):
    # 设定筛选时间范围
    start_date = pd.Timestamp("2014-01-01 11:59:00+00:00")
    end_date = pd.Timestamp('2023-12-31 11:59:00+00:00')
    df = df.loc[start_date:end_date]
    # # #去趋势
    # df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    # 3倍中误差去噪
    mse = 3 * np.mean(df.iloc[:, 1])
    df.loc[df.iloc[:, 1] > mse, :] = np.nan

    # 重新排列好时期index
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_range)
    df = df.fillna(method='bfill')
    # 采用三次样条填充缺失值
    df.iloc[:, 0].interpolate(method='spline', order=3, inplace=True)
    df.iloc[:, 1].interpolate(method='spline', order=3, inplace=True)
    # 同震修正
    df.iloc[1047:, 0] = df.iloc[1047:, 0] - (df.iloc[1047, 0] - df.iloc[1046, 0])
    df.iloc[1048:, 0] = df.iloc[1048:, 0] - (df.iloc[1048, 0] - df.iloc[1047, 0])
    df.iloc[1049:, 0] = df.iloc[1049:, 0] - (df.iloc[1049, 0] - df.iloc[1048, 0])
    #去趋势
    df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    #滤波
    L = 60
    fc = 1 / 50
    nyquist = 0.5
    normalized_cutoff = fc / nyquist
    FIR = firwin(L, normalized_cutoff, window='hamming')
    df_filter = np.convolve(df.iloc[:, 0], FIR, mode='same')
    df.iloc[:, 0] = df_filter
    if type == 'vel':
        #差分
        df.iloc[1:, 0] = np.diff(df.iloc[:, 0])
        df.iloc[0, 0] = df.iloc[1, 0]
        return df
    elif type == 'disp':
        return df

def preprocess_deep(df,type):
    # 设定筛选时间范围
    start_date = pd.Timestamp("2014-01-01 11:59:00+00:00")
    end_date = pd.Timestamp('2023-12-31 11:59:00+00:00')
    df = df.loc[start_date:end_date]
    # # #去趋势
    # df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    # 3倍中误差去噪
    mse = 3 * np.mean(df.iloc[:, 1])
    df.loc[df.iloc[:, 1] > mse, :] = np.nan

    # 重新排列好时期index
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_range)
    df = df.fillna(method='bfill')
    # 采用三次样条填充缺失值
    df.iloc[:, 0].interpolate(method='spline', order=3, inplace=True)
    df.iloc[:, 1].interpolate(method='spline', order=3, inplace=True)
    #同震修正
    df.iloc[1047:, 0] = df.iloc[1047:, 0] - (df.iloc[1047, 0] - df.iloc[1046, 0])
    df.iloc[1048:,0]=df.iloc[1048:,0]-(df.iloc[1048, 0]-df.iloc[1047, 0])
    df.iloc[1049:, 0] = df.iloc[1049:, 0] - (df.iloc[1049, 0] - df.iloc[1048, 0])
    # df.iloc[1057:, 0] = df.iloc[1057:, 0] - (df.iloc[1057, 0] - df.iloc[1056, 0])
    # df.iloc[1058:, 0] = df.iloc[1058:, 0] - (df.iloc[1058, 0] - df.iloc[1057, 0])
    #去趋势
    df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    # #滤波
    # L = 60
    # fc = 1 / 50
    # nyquist = 0.5
    # normalized_cutoff = fc / nyquist
    # FIR = firwin(L, normalized_cutoff, window='hamming')
    # df_filter = np.convolve(df.iloc[:, 0], FIR, mode='same')
    # df.iloc[:, 0] = df_filter
    if type == 'vel':
        #差分
        df.iloc[1:, 0] = np.diff(df.iloc[:, 0])
        df.iloc[0, 0] = df.iloc[1, 0]
        return df
    elif type == 'disp':
        return df

def preprocess_nofilter(df,type):
    # 设定筛选时间范围
    start_date = pd.Timestamp("2014-01-01 11:59:00+00:00")
    end_date = pd.Timestamp('2023-12-31 11:59:00+00:00')
    df = df.loc[start_date:end_date]
    # # #去趋势
    # df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    # 3倍中误差去噪
    mse = 3 * np.mean(df.iloc[:, 1])
    df.loc[df.iloc[:, 1] > mse, :] = np.nan

    # 重新排列好时期index
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_range)
    df = df.fillna(method='bfill')
    # 采用三次样条填充缺失值
    df.iloc[:, 0].interpolate(method='spline', order=3, inplace=True)
    df.iloc[:, 1].interpolate(method='spline', order=3, inplace=True)
    # 同震修正
    df.iloc[1047:, 0] = df.iloc[1047:, 0] - (df.iloc[1047, 0] - df.iloc[1046, 0])
    df.iloc[1048:, 0] = df.iloc[1048:, 0] - (df.iloc[1048, 0] - df.iloc[1047, 0])
    df.iloc[1049:, 0] = df.iloc[1049:, 0] - (df.iloc[1049, 0] - df.iloc[1048, 0])
    #去趋势
    df.iloc[:, 0] = signal.detrend(df.iloc[:, 0], type='linear')
    # #滤波
    # L = 60
    # fc = 1 / 50
    # nyquist = 0.5
    # normalized_cutoff = fc / nyquist
    # FIR = firwin(L, normalized_cutoff, window='hamming')
    # df_filter = np.convolve(df.iloc[:, 0], FIR, mode='same')
    # df.iloc[:, 0] = df_filter
    if type == 'vel':
        #差分
        df.iloc[1:, 0] = np.diff(df.iloc[:, 0])
        df.iloc[0, 0] = df.iloc[1, 0]
        return df
    elif type == 'disp':
        return df

def filter(df):
    #滤波
    L = 100
    fc = 1 / 50
    nyquist = 0.5
    normalized_cutoff = fc / nyquist
    FIR = firwin(L, normalized_cutoff, window='hamming')
    df_filter = np.convolve(df, FIR, mode='same')
    return df_filter


# 延迟嵌入函数
def delay_embedding_and_svd_decomponent(data):
    # 参数设置
    delay = 1  # 延迟
    embedding_dim = 50  # 嵌入向量的维度
    data = np.diff(filter(data))

    n_samples = len(data) - (embedding_dim - 1) * delay
    if n_samples <= 0:
        raise ValueError("输入时间序列长度不足以进行延迟嵌入")

    # 构造嵌入矩阵
    embedded_matrix = np.array([data[i:i + n_samples] for i in range(0, embedding_dim * delay, delay)]).T

    U, S, Vt = svd(embedded_matrix, full_matrices=False)

    return U


