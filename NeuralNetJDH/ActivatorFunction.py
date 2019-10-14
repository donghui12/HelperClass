import numpy as np


# 激活函数， 计算前向传播函数，后向传播函数
class Aativator(object):
    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        pass


# 直传函数，相当于无激活
class Identity(Aativator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta, a


class Sigmoid(Aativator):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        detal_A = np.multiarray(a, 1-a)
        delta_Z = np.multiarray(delta, detal_A)
        return delta_Z, detal_A


class Tanh(Aativator):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        delta_A = 1 - np.multiarray(a, a)
        delta_Z = np.multiarray(delta, delta_A)
        return delta_Z, delta_A


class Relu(Aativator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        delta_A = np.zeros(z.shape)
        delta[z > 0] = 1
        delta_Z = delta_A*delta
        return delta_Z, delta_A
