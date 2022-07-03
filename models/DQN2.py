import torch.nn as nn

class DQN2(nn.Module):
    def __init__(self):
        super().__init__()

        #this could be one block but idc 
        self.l1 = nn.Sequential(nn.Linear(4,69), nn.ReLU()) #add whatever hidden layers ig
        self.l2 = nn.Sequential(nn.Linear(69, 420), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(420, 1))

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x