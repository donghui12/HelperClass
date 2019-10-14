import numpy as np
from pathlib import Path

from NeuralNetJDH.NetType import InitialMethod


class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta
        # 初始化文件夹
        self.initial_value_filename = str.format("w_{0}_{1}_{2}_init", self.num_input, self.num_output,
                                                 self.init_method.name)

    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        self.delta_W = np.zeros(self.W.shape)
        self.delta_b = np.zeros(self.b.shape)

    def __CreateNew(self):
        # 创建新的文件夹
        self.W, self.b = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()

    def __LoadExistingParameters(self):
        # 加载已存在的文件
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        w_file = Path(file_name)
        if w_file.exists():
            self.__LoadExistingParameters()
        else:
            self.__CreateNew()

    def update(self):
        self.W = self.W - self.eta * self.delta_W
        self.b = self.b - self.eta * self.delta_b

    def __SaveInitialValue(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        np.savez(file_name, weights=self.W, bias=self.b)

    def __LoadInitialValue(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    def SaveResultValue(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        np.savez(file_name, weights=self.W, bias=self.b)

    def LoadResultValue(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        # 初始化 W,b
        if method == InitialMethod.Zero:
            W = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            W = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_input, num_output))
        b = np.zeros((1, num_output))
        return W, b