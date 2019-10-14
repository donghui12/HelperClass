from NeuralNetJDH.NetType import NetType
from NeuralNetJDH.NetType import InitialMethod


class HyperParameters(object):
    def __init__(self, n_input, n_hidden, n_output, eta=0.1, max_epoch=10000, net_type=NetType.Fitting,
                 init_method=InitialMethod.Xavier):
        """
            net_type: choose the type of net, such as  BinaryClassifier, MultipleClassifier
            init_method : choose the number of plier
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.eta = eta
        self.max_epoch = max_epoch
        self.net_type = net_type
        self.init_method = init_method

    def toString(self):
        title = str.format("bz:{0},eta:{1},ne:{2}", self.batch_size, self.eta, self.num_hidden)
        return title