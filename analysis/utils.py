import numpy as np

def averaged_by_interval(time_series, interval):
    return np.mean(time_series.reshape(-1, interval), axis=1)

def calculate_length_of_day(jd):
    jd = np.array(jd)
    len_of_day = np.zeros((len(jd)-1)//2)
    for j in range(0, len(jd)-1, 2):
        len_of_day[j//2] = jd[j+1] - jd[j]
    return len_of_day


def split_data(data):
    # data = data.tolist()
    s01, s02 = [], []
    chunk01 = data[:77]
    s01 = np.concatenate([s01,chunk01])
    for i in range(len(data) // 365 + 100):
        chunk1 = data[i * 365 + 265:i * 365 + 265 + 100 + 77]
        s01 = np.concatenate([s01,chunk1])
    for i in range(len(data) // 365):
        chunk2 = data[i * 365 + 77:i * 365 + 265]
        s02 = np.concatenate([s02,chunk2])
    s1 = np.array(s01)
    s2 = np.array(s02)
    return s1, s2