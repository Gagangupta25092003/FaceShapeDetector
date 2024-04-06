import torch

class Classifier(torch.nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.linear1 = torch.nn.Linear(478*2, 524)
        self.linear2 = torch.nn.Linear(524, 128)
        self.linear3 = torch.nn.Linear(128, 5)
        self.activation = torch.nn.ReLU()
        self.batchNormalization1 = torch.nn.BatchNorm1d(524)        
        self.batchNormalization2 = torch.nn.BatchNorm1d(128)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.batchNormalization1(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.batchNormalization2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x