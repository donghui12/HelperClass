### NeuralNet 需要的辅助函数
撰写这些函数的目的是建立自己的一套神经网络体系  
首先能适用大部分的环境  
其次要符合软件工程的基本要求（开闭原则）  
最后是易于调试
#### 1. 加载数据  
    class DataReader(object):
        def __init__(self, train_file, test_file):
            self.train_file_name = train_file
            self.test_file_name = test_file
            self.num_train = 0        # num of training examples
            self.num_test = 0         # num of test examples
            self.num_validation = 0   # num of validation examples
            self.num_feature = 0      # num of features
            self.num_category = 0     # num of categories
            self.XTrain = None        # training feature set
            self.YTrain = None        # training label set
            self.XTest = None         # test feature set
            self.YTest = None         # test label set
            self.XTrainRaw = None     # training feature set before normalization
            self.YTrainRaw = None     # training label set before normalization
            self.XTestRaw = None      # test feature set before normalization
            self.YTestRaw = None      # test label set before normalization
            self.XDev = None          # validation feature set
            self.YDev = None          # validation lable set
        def ReadData(self):
            # read data from file
            pass
        def NormalizeX(self):
            # Normalize X
            pass
        def NormalizeY(self):
            # Normalize Y
            pass
        def DeNormalizeY(self, predict_data):
            # DeNormalize the result of Predicate Data
            pass
        def NormalizePredicateData(self):
            # Normalize Predicate Data
            pass
        def __ToOneHot(self, Y, base=0):
            # for binary classifier
            pass
 加载数据，以及对数据进行预处理（归一化）
#### 2. 设置参数
    from enum import Enum
    class NetType(Enum):
        Fitting = 1,
        BinaryClassifier = 2,
        MultipleClassifier = 3
    
    class InitialMethod(Enum):
        Zero = 0,
        Normal = 1,
        Xavier = 2,
        MSRA = 3

#### 3. 初始化参数  
    class HyperParameters(object):
        def __init__(self, n_input, n_hidden, n_output, 
                    eta=0.1, max_epoch=10000, net_type = NetType.Fitting
                    init_method = init_method):
            """
                net_type: choose the type of net, such as  BinaryClassifier, MultipleClassifier
                init_method : choose the number of plier
            """
            self.n_input = n_input
            ...
            ...
            self.init_method = init_method
        def toString(self):
            title = str.format("bz:{0},eta:{1},ne:{2}", self.batch_size, self.eta, self.num_hidden)
            return title
添加超参数对神经网络的修改
        
#### 4. 激活函数  
    class Aativator(object):
        def forward(self, z):
            pass
        def backward(self, z, a, delta):
            pass
    
    class Identity(Aativator):
        def forward(self, z):
            return z
        def backward(self, z, a,delta):
            return delta, a
在激活函数中：Aativator是激活函数普遍类型  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Identity(Aativator)是直传函数，相当于无激活
激活函数还有：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sigmoid(Aativator)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Tanh(CActivator)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Relu(CActivator)  
等等，需要根据不同功能选择
#### 5. 分类函数  
    class CClassifier(object):
        def forward(self, z):
            pass    
    class Logistic(CClassifier):
        def forward(self, z):
            a = 1.0 / (1.0 + np.exp(-z))
            return a
    class Softmax(CClassifier):
        def forward(self, z):
            shift_z = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(shift_z)
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return a

分类函数是适用于分类时，最后一步要加这个函数实现分类  
与激活函数不同，激活函数是每一层结束都要加激活函数，  
但最后一层不需要激活函数

#### 6. 损失函数（loss function）
    class LossFunction(object):
        def __init__(self, net_type):
            self.net_type = net_type
            
        # fcFunc: feed forward calculation
        def CheckLoss(self, A, Y):
            m = Y.shape[0]
            if self.net_type == NetType.Fitting:
                loss = self.MSE(A, Y, m)
            elif self.net_type == NetType.BinaryClassifier:
                loss = self.CE2(A, Y, m)
            elif self.net_type == NetType.MultipleClassifier:
                loss = self.CE3(A, Y, m)
            #end if
            return loss
        def MSE(self, A, Y, count):
            pass
        # end def

        # for binary classifier
        def CE2(self, A, Y, count):
            pass
        # end def
    
        # for multiple classifier
        def CE3(self, A, Y, count):
            pass
        # end def
        
#### 7. 可视化数据
    import matplotlib.pyplot as plt
    import pickle
    class TrainingHistory(object):
            def __init__(self):
                self.loss_train = []
                self.accuracy_train = []
                self.iteration_seq = []
                self.epoch_seq = []
                self.loss_val = []
                self.accuracy_val = []
  
            def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
                self.iteration_seq.append(total_iteration)
                self.epoch_seq.append(epoch)
                self.loss_train.append(loss_train)
                self.accuracy_train.append(accuracy_train)
                if loss_vld is not None:
                    self.loss_val.append(loss_vld)
                if accuracy_vld is not None:
                    self.accuracy_val.append(accuracy_vld)
        
                return False
        
            # 图形显示损失函数值历史记录
            def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
                fig = plt.figure(figsize=(12,5))
        
                axes = plt.subplot(1,2,1)
                #p2, = axes.plot(self.iteration_seq, self.loss_train)
                #p1, = axes.plot(self.iteration_seq, self.loss_val)
                p2, = axes.plot(self.epoch_seq, self.loss_train)
                p1, = axes.plot(self.epoch_seq, self.loss_val)
                axes.legend([p1,p2], ["validation","train"])
                axes.set_title("Loss")
                axes.set_ylabel("loss")
                axes.set_xlabel("epoch")
                if xmin != None or xmax != None or ymin != None or ymax != None:
                    axes.axis([xmin, xmax, ymin, ymax])
                
                axes = plt.subplot(1,2,2)
                #p2, = axes.plot(self.iteration_seq, self.accuracy_train)
                #p1, = axes.plot(self.iteration_seq, self.accuracy_val)
                p2, = axes.plot(self.epoch_seq, self.accuracy_train)
                p1, = axes.plot(self.epoch_seq, self.accuracy_val)
                axes.legend([p1,p2], ["validation","train"])
                axes.set_title("Accuracy")
                axes.set_ylabel("accuracy")
                axes.set_xlabel("epoch")
                
                title = params.toString()
                plt.suptitle(title)
                plt.show()
                return title
        
            def ShowLossHistory4(self, axes, params, xmin=None, xmax=None, ymin=None, ymax=None):
                p2, = axes.plot(self.epoch_seq, self.loss_train)
                p1, = axes.plot(self.epoch_seq, self.loss_val)
                title = params.toString()
                axes.set_title(title)
                axes.set_xlabel("epoch")
                axes.set_ylabel("loss")
                if xmin != None and ymin != None:
                    axes.axis([xmin, xmax, ymin, ymax])
                return title
        
            def Dump(self, file_name):
                f = open(file_name, 'wb')
                pickle.dump(self, f)
        
            def Load(file_name):
                f = open(file_name, 'rb')
                lh = pickle.load(f)
                return lh

#### 8. 神经网络建立
    class NeuralNet(object):
        def __init__(self, params):
            self.params = params  # HyperParameters
            self.wb = WeightsBias(self.params.num_input, self.params.num_hidden, self.params.init_method, self.params.eta)
            self.subfolder = os.getcwd()+'\\'+self.__create_subfolder()
            # the folder is the location of saving result 
        def __create_subfolder(self):
            pass
        def forward(self, batch_X):
            pass
        def backward(self, batch_X, batch_Y, batch_A):
            pass
        def update(self):
            pass
        def infernce(self, x):
            # x is the Predicate Sets
            pass
        def train(self, dataReader, need_test):
            pass
        def SaveResult(self):
            pass
        def loadResult(self):
            pass


调用顺序：
首先-->设置参数-->加载数据-->初始化超参数-->建立神经网络-->训练-->测试-->可视化
其他辅助函数都是在神经网络中使用到的
        