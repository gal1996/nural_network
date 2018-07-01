import sys, os
sys.path.append(os.pardir)
import ActivationFunc as af
import LossFunc as lf
import Gradient as gd
import numpy as np
import layers as lay
from collections import OrderedDict

class two_layer_net:
    #重みの初期化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #レイヤーの作成
        self.layers = OrderedDict()
        self.layers["affine1"] = lay.affin_layer(self.params['W1'], self.params['b1']) #インスタンス生成時の引数はコンストラクタへの引数
        self.layers["relu1"] = lay.relu_layer()
        self.layers["affine2"] = lay.affin_layer(self.params['W2'], self.params['b2'])

        self.lastlayer = lay.softmax_loss_layer()

    #値の推定を行う
    def predict(self, x):
        for layer in self.layers.values():
            #ここはxにしないと次の層に値をうまく渡せない
            x = layer.forward(x)

        return x

    #損失関数の値を出す
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastlayer.forward(y, t)

    #正確度を計算する
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    #勾配を計算する
    def cal_gradient(self, x, t):
        #まず損失関数の値を得る
        self.loss(x, t)

        #各層から逆伝搬をしていく
        dout = 1
        dout = self.lastlayer.backward(dout)

        #self.layersの順番は変えられないから、そのリストを複製してそれを逆順に並べる
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #WとBの微分はaffinレイヤのインスタンス変数として格納されているの
        grad = {}

        grad['W1'] = self.layers["affine1"].dW
        grad['b1'] = self.layers["affine1"].dB
        grad['W2'] = self.layers["affine2"].dW
        grad['b2'] = self.layers["affine2"].dB

        return grad
