from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor as Regressor

from dataset import TransDataset
import numpy as np
trans_dataset = TransDataset()
X, y = trans_dataset.get_x_y()


regressor = Regressor(random_state=0, max_depth=2)

regressor = regressor.fit(X, y)

test_dataset = TransDataset("test")
test_x, test_y = test_dataset.get_x_y()

total_count, count, total_loss = 0, 0, 0


for x, y in zip(test_x, test_y):
    loss = abs(regressor.predict([x]) - y[0])
    total_loss += loss
    total_count += 1
    if loss > 50:
        count += 1
print("total count", total_count)
print("cross the boundary count", count)
print("cross rate", count/total_count)
print("average lss", total_loss/total_count)

