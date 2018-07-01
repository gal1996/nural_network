import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import two_layer_net
from matplotlib import pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

#ハイパーパラメタの設定
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

n_net = two_layer_net(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #バッチデータの取得
    batch_id = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_id]
    t_batch = t_train[batch_id]

    #勾配の計算
    grad = n_net.cal_gradient(x_batch, t_batch)

    #パラメータの更新
    for i in ('W1', 'b1', 'W2', 'b2'):
        n_net.params[i] -= learning_rate * grad[i]

    loss = n_net.loss(x_batch, t_batch)
    print(loss)
    train_loss_list.append(loss)

#損失関数の値のプロット
x = range(iters_num)
y = train_loss_list

plt.plot(x,y)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
