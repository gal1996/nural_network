import numpy as np
import ActivationFunc as ac
import LossFunc as lf

class mul_layer:
    def __init__(self):
        self.x = None
        self.y = None

    #層を前に進める関数
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*Y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class add_layer:
    def __init__(self):
            pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class relu_layer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        #xがx<=0以上のインデックス番号にtrueが入っている
        out = x.copy()
        self.mask = (x <= 0)
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class sigmoid_layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.x = x
        self.y = out

        return out

    def backward(self, dout):
        dx = dout * (self.y**2) * np.exp(-self.x)

        return dx

class affin_layer:
    def __init__(self, W, B):
        self.X = None
        self.dW = None
        self.dB = None
        self.W = W
        self.B = B

    def forward(self, X):
        self.X = X
        XW = np.dot(self.X, self.W)
        out = XW + self.B

        return out

    def backward(self, dout):
        dY = dout
        dX = np.dot(dY, self.W.T)
        self.dW = np.dot(self.X.T, dY)
        self.dB = np.sum(dY, axis = 0)

        return dX

class softmax_loss_layer:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = ac.softmax(x)
        self.loss = lf.cross_entropy_error(self.y, self.t)

        out = self.loss

        return out

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
