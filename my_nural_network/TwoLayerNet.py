import sys, os
sys.path.append(os.pardir)
import ActivationFunc as af
import LossFunc as lf
import Gradient as gd
import numpy as np

class two_layer_net:
    #重みの初期化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #値の推定を行う
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = af.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = af.softmax(a2)

        return y

    #損失関数の値を出す
    def loss(self, x, t):
        y = self.predict(x)

        return lf.cross_entropy_error(y, t)

    #正確度を計算する
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    #勾配を計算する
    def cal_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grad = {}
        grad['W1'] = gd.numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = gd.numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = gd.numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = gd.numerical_gradient(loss_W, self.params['b2'])

        return grad
