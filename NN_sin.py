import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NeuralNetJDH.NeuralNet import *
from NeuralNetJDH.DataReader import *

train_data_name = './datasets/npz/ch08.train.npz'
test_data_name = './datasets/npz/ch08.test.npz'


def ShowResult():
    pass


if __name__ == "__main__":
    dataReader = DataReader(train_data_name, test_data_name)

    n_input, n_hidden, n_output = 1, 2, 1
    eta, max_epoch = 0.05, 10000

    params = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet(params, 'sin_121')

    net.train(dataReader, True)
    # ShowResult(net, dataReader, params.toString())