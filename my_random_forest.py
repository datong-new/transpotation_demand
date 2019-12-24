from sklearn import tree
from dataset import TransDataset
import numpy as np
import random

class MyRandomForest():
    def __init__(self, trees_num=2):
        self.trans_dataset = TransDataset()
        self.X, self.y = self.trans_dataset.get_x_y()
        self.len = self.X.shape[0]
        self.regressors = []
        self.trees_num = trees_num
        self.construct_regressors()

    def construct_decision_regressor(self):
        new_X, new_y = self.get_newx_newy()
        regressor = tree.DecisionTreeRegressor()
        regressor = regressor.fit(new_X, new_y)
        self.regressors.append(regressor)

    def construct_regressors(self):
        for i in range(self.trees_num):
            self.construct_decision_regressor()

    def get_newx_newy(self):
        new_x, new_y = [], []
        for i in range(self.len):
            index = int(random.uniform(0, self.len))
            new_x.append(self.X[index])
            new_y.append(self.y[index])
        return np.array(new_x, dtype=np.float), np.array(new_y, dtype=np.float)
        
    def predict(self, x):
        y = 0
        for regressor in self.regressors:
            y += regressor.predict([x])
        return y/len(self.regressors)
            


random_forest = MyRandomForest(trees_num=30)

test_dataset = TransDataset("test")
test_x, test_y = test_dataset.get_x_y()

total_count, count, total_loss = 0, 0, 0
print("len test_x", len(test_x))

for x, y in zip(test_x, test_y):
    total_count += 1
    loss = abs(random_forest.predict(x)[0] - y[0])
    total_loss += loss
    if loss > 100:
        count += 1
print("total count", total_count)
print("cross the boundary count", count)
print("cross rate", count/total_count)

print("average loss", total_loss/total_count)
