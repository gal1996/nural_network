import numpy as np

#勾配を求める関数を持つクラス
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    if(x.ndim == 1):
        x = x.reshape(1, x.size)

    grad = np.zeros_like(x)
    #print('x.size' + str(x.size))
    #print('x.shpe[0]' + str(x.shape[0]))
    #print(x[1])


    for idx in range(x.shape[0]):
        for idy in range(x.shape[1]):
            tmp_val = x[idx][idy]
            x[idx][idy] = tmp_val + h
            fxh1 = f(x) # f(x+h)

            x[idx][idy] = tmp_val - h
            fxh2 = f(x) # f(x-h)
            grad[idx][idy] = float((fxh1 - fxh2) / (2*h))

            x[idx][idy] = tmp_val # 値を元に戻す

    if(grad.shape == (1, x.size)):
        grad = grad.reshape(-1)

    return grad
