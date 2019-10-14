import csv
import numpy as np
from pathlib import Path


class DataReader(object):
    def __init__(self, train_file, test_file):
        self.train_file_name = train_file  # the path of train_file
        self.test_file_name = test_file  # the path of test file
        self.num_train = 0  # num of training examples
        self.num_test = 0  # num of test examples
        self.num_validation = 0  # num of validation examples
        self.num_feature = 0  # num of features
        self.num_category = 0  # num of categories
        self.XTrain = None  # training feature set
        self.YTrain = None  # training label set
        self.XTest = None  # test feature set
        self.YTest = None  # test label set
        self.XTrainRaw = None  # training feature set before normalization
        self.YTrainRaw = None  # training label set before normalization
        self.XTestRaw = None  # test feature set before normalization
        self.YTestRaw = None  # test label set before normalization
        self.XDev = None  # validation feature set
        self.YDev = None  # validation lable set

    def Read_CSV_Data(self):
        # read data from CSV file
        train_file = Path(self.train_file_name)
        if train_file.exists():
            with open(self.train_file_name) as f:
                csv_reader = csv.reader(f)
                rows = [row for row in csv_reader]
                self.XTrainRaw = np.array([row[:-1] for row in rows], dtype=np.float32)
                self.YTrainRaw = np.array([row[-1] for row in rows], dtype=np.float32)
        else:
            raise Exception('connot find train file!!!')

    def Read_NPZ_Data(self):
        # read data from NPZ file
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrainRaw = data['data']
            self.YTrainRaw = data['label']
            assert(self.XTrainRaw.shape[0] == self.YTrainRaw.shape[0])
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = len(np.unique(self.YTrainRaw))
        else:
            raise Exception('connot find train file!!!')

    def NormalizeX(self, raw_data):
        # Normalize X
        min_value = np.min(self.XTrainRaw, axis=0)
        max_value = np.max(self.XTrainRaw, axis=0)
        self.X_norm = np.vstack((min_value, max_value))
        temp_X = (self.XTrainRaw - min_value) / (max_value - min_value)

        return temp_X

    def NormalizeY(self, raw_data):
        # Normalize Y
        min_value = np.min(self.YTrainRaw, axis=0)
        max_value = np.max(self.YTrainRaw, axis=0)
        self.Y_norm = np.vstack((min_value, max_value))
        temp_Y = (self.YTrainRaw - min_value) / (max_value - min_value)

        return temp_Y

    def DeNormalizeY(self, predict_data):
        # DeNormalize the result of Predicate Data
        real_value = predict_data * self.Y_norm[1, 0] + self.Y_norm[0, 0]
        return real_value

    def NormalizePredicateData(self, PredicateData):
        # Normalize Predicate Data
        X_new = np.zeros(PredicateData.shape)
        n_feature = PredicateData.shape[0]
        for i in range(n_feature):
            x = PredicateData[i, :]
            X_new[i, :] = (x - self.X_norm[0, i])/self.X_norm[1, i]
        return X_new

    def __ToOneHot(self, Y, base=0):
        # for binary classifier
        count = Y.shape[0]
        temp_Y = np.zeros((count, self.num_category))
        for i in range(count):
            n = int(Y[i, 0])
            temp_Y[i, n-base] = 1
        return temp_Y
