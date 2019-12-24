from scipy.io import loadmat
import random

content = loadmat("demand_10.mat")["demand"]
date=-1
for i in range(content.shape[0]):
    print(i)
    if i//144 == i/144:
        date += 1
        date = date % 7
    row = content[i]
    idx = i % 144
    if random.uniform(0,1) < 0.2:
        f = open("data_test.txt", "a")
    else:
        f = open("data_train.txt", "a")

    for col, y in enumerate(row):
        f.write("{}\t{}\t{}\t{}\n".format(date, idx, col, y)) # date, sample t, region

    f.close()
