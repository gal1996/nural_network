import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet_ver2 import two_layer_net
from matplotlib import pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#ハイパーパラメタの設定
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)


n_net = two_layer_net(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #バッチデータの取得
    batch_id = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_id]
    t_batch = t_train[batch_id]

    #勾配の計算
    grad = n_net.cal_gradient(x_batch, t_batch)

    #パラメータの更新
    for j in ('W1', 'b1', 'W2', 'b2'):
        n_net.params[j] -= learning_rate * grad[j]

    loss = n_net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0 :
        train_acc = n_net.accuracy(x_train, t_train)
        test_acc = n_net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | "  +str(train_acc) + ", " + str(test_acc))

#損失関数の値のプロット
x = range(iters_num)
y = train_loss_list

plt.plot(x,y)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
