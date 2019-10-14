from NeuralNetJDH.WeightsBias import WeightsBias
from NeuralNetJDH.LossFunction import LossFunction
from NeuralNetJDH.ActivatorFunction import *
from NeuralNetJDH.NetType import *
from NeuralNetJDH.DataReader import *
from NeuralNetJDH.ClassifierFunction import *
from NeuralNetJDH.HyperParameters import *
from NeuralNetJDH.TrainingHistory import *

import os


class NeuralNet(object):
    def __init__(self, params, model_name):
        self.params = params  # HyperParameters
        self.wb = WeightsBias(self.params.num_input, self.params.num_hidden, self.params.init_method, self.params.eta)
        self.model_name = model_name
        self.subfolder = os.getcwd() + '\\' + self.__create_subfolder()
        # the folder is the location of saving result
        # 初始化wb1, wb2
        self.wb1 = WeightsBias(self.params.num_input, self.params.num_hidden, self.params.init_method, self.params.eta)
        self.wb1.InitializeWeights(self.subfolder, True)
        self.wb2 = WeightsBias(self.params.num_hidden, self.params.num_output, self.params.init_method, self.params.eta)
        self.wb2.InitializeWeights(self.subfolder, True)

    def __create_subfolder(self):
        if self.model_name is not None:
            path = self.model_name.strip()
            path = path.rstrip('\\')
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self, batch_X):
        # layer 1
        self.Z1 = np.dot(batch_X, self.wb1.W) + self.wb1.b
        self.A1 = Sigmoid().forward(self.Z1)

        # layer 2
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.b
        if self.params.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.params.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)
        else:
            self.A2 = self.Z2

        self.output = self.A2
        return self.output

    def backward(self, batch_X, batch_Y, batch_A):
        # layer 1
        m = batch_X.shape[0]
        delta_Z2 = self.A2 - batch_Y
        self.wb2.delta_W = np.dot(self.A1.T, delta_Z2) / m
        self.wb2.delta_b = np.sum(delta_Z2, axis=0, keepdims=True) / m

        delta_1 = np.dot(delta_Z2, self.wb2.W.T)
        delta_Z1, _ = Sigmoid().backward(None, self.A1, delta_1)
        self.wb1.delta_W = np.dot(batch_X.T, delta_Z1) / m
        self.wb1.delta_b = np.sum(delta_Z1, axis=0, keepdims=True) / m

    def update(self):
        self.wb1.update()
        self.wb2.update()

    def inference(self, x):
        # x is the Predicate Sets
        self.backward(x)
        return self.output

    def train(self, dataReader, need_test):
        # dataReader = DataReader()
        dataReader.Read_NPZ_Data()
        batch_x = dataReader.NormalizeX(1)
        batch_y = dataReader.NormalizeY(1)
        for epoch in range(int(self.params.max_epoch)):
            if epoch % 1000 == 0:
                print('epoch=%d' %epoch)
            batch_a = self.forward(batch_X=batch_x)
            self.backward(batch_x, batch_y, batch_a)
            self.update()
        self.SaveResult()

    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder, 'wb1')
        self.wb2.SaveResultValue(self.subfolder, 'wb2')

    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder, 'wb1')
        self.wb2.LoadResultValue(self.subfolder, 'wb2')

    def __CalAccuracy(self, a, y):
        # 计算准确率
        assert(a.shape == y.shape)
        m = a.shape[0]
        if self.params.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a-y)**2)/m
            r2 = 1 - mse/var
            return r2
        elif self.params.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            correct = r.sum()
            return correct/m
        elif self.params.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct/m
