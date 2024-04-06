import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

training_data = np.genfromtxt('Training_data.csv', delimiter=',', dtype=float)
training_label = np.genfromtxt('Training_data.csv', delimiter=',', dtype=float)
testing_data = np.genfromtxt('Testing_data.csv', delimiter=',', dtype=float)
testing_label = np.genfromtxt('Testing_label.csv', delimiter=',', dtype=float)

train = CustomDataset(training_data, training_label)
test = CustomDataset(testing_data, testing_label)