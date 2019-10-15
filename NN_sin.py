import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NeuralNetJDH.NeuralNet import *
from NeuralNetJDH.DataReader import *

train_data_name = './datasets/npz/ch09.train.npz'
test_data_name = './datasets/npz/ch09.test.npz'


def ShowLoss(loss, max_epoch):
    y = loss
    x = list(range(max_epoch))
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Time(s)")  # X轴标签
    plt.ylabel("Volt")  # Y轴标签
    plt.title("Line plot")  # 图标题
    plt.show()  # 显示图
    plt.savefig("line.jpg")  # 保存图


def ShowResult(net, dataReader, title):
    X, Y = dataReader.NormalizeX(1), dataReader.NormalizeY(1)
    plt.plot(X[:, 0], Y[:, 0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0, 1, 100).reshape(100, 1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.Read_NPZ_Data()
    n_input, n_hidden, n_output = 1, 3, 1
    eta, max_epoch = 0.5, 1000000

    params = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet(params, 'complex')

    net.train(dataReader, True)
    loss = net.ShowLoss()
    ShowLoss(loss, max_epoch)
    ShowResult(net, dataReader, params.toString())