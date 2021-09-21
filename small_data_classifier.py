import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallDataClassifier(nn.Module):
    
    def __init__(self, sparse_layer, inputs, outputs, print_act_shape=False):
        super().__init__()

        self.print_act_shape = print_act_shape
        self.sparse_layer = sparse_layer
        
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(inputs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, outputs)

    # x represents our data
    def forward(self, x):
        activations = self.sparse_layer(x)
        x = torch.flatten(activations, 1)
        
        if self.print_act_shape:
            print(x.shape)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, activations