import torch.utils.data as data
import random
import numpy as np

class TransDataset(data.Dataset):
    
    def __init__(self,  type="train", K=10):
        super(TransDataset, self).__init__()
        self.type = type
        path = "./data_train.txt" if self.type == "train" else "./data_test.txt"
        self.datas = self.get_datas(path)
        if self.type == "test":
            self.datas = np.array(self.datas)
            return
        
        self.datas = np.array(self.datas)
        """ for cross validation, but never be used in the experiment
        vals, trains = [], []
        for i in range(len(self.datas)):
            if random.uniform(0,1) < 1/K:
                vals.append(i)
            else:
                trains.append(i)
        if self.type == "train":
            self.datas = np.array(self.datas)[trains]
        else:
            self.datas = np.array(self.datas)[vals]
        """
    def get_x_y(self):
        return self.datas[:, :3], self.datas[:, 3:4]
        
    def get_datas(self, path):
        with open(path, "r") as f:
            datas = []
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                data = line.split("\t")
                data = [float(item) for item in data]
                datas.append(data)
        return datas

    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):
        return np.array(self.datas[index][:2],dtype=np.float), np.array(self.datas[index][3:4], dtype=np.float)

