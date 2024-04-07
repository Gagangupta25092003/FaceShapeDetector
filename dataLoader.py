import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

training_data = np.genfromtxt('Training_data.csv', delimiter=',', dtype=float).reshape((3996, 478*2))
training_label = np.genfromtxt('Training_labels.csv', delimiter=',', dtype=float).reshape((3996,))
testing_data = np.genfromtxt('Testing_data.csv', delimiter=',', dtype=float).reshape((1000, 478*2))
testing_label = np.genfromtxt('Testing_labels.csv', delimiter=',', dtype=float).reshape((1000,))

training_data = training_data.reshape((3996, 478, 2))
for i in range(3996):
    if random.randint(0,1) == 1:
        training_data[i, :, 1]*=1
training_data = training_data.reshape((3996, 478*2))

train = CustomDataset(training_data, training_label)
test = CustomDataset(testing_data, testing_label)