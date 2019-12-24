from sklearn import tree
from sklearn.ensemble import RandomForestRegressor as Regressor

from dataset import TransDataset
import numpy as np
trans_dataset = TransDataset()
X, y = trans_dataset.get_x_y()
y = [item[0] for item in y]

regressor = Regressor(random_state=0, max_depth=2)
regressor = regressor.fit(X, y)

test_dataset = TransDataset("test")
test_x, test_y = test_dataset.get_x_y()

total_count, count = 0, 0
print("len test_x", len(test_x))


for x, y in zip(test_x, test_y):
    total_count += 1
    if abs(regressor.predict([x]) - y[0]) > 50:
        count += 1
print(total_count)
print(count)

