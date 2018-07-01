import numpy as np

#(xはデータ,tはone-hot表現の教師データ)
#２乗和誤差
def mean_squared_error(x, t):
    return 0.5 * np.sum((x - t)**2)
"""
#交差エントロピー誤差
def cross_entropy_error(x, t):
    delta = 1e-7 #xが0であったときエラーになるのを防ぐため
    if(x.ndim == 1):
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    batch_size = x.shape[0]
# 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == x.size:
        t = t.argmax(axis=1)


    return -np.sum(np.log(x[np.arange(batch_size), t] + delta)) / batch_size
"""

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
