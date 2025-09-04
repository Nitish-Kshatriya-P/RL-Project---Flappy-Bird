import torch
from torch import nn
import torch.nn.functional as F 

class DQN(nn.Module):
    def __init__(self, start_dim, action_dim, hidden_dim = 512):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
if __name__ == '__main__':
    start_dim = 12
    action_dim = 2
    network = DQN(start_dim,action_dim)
    rand_input = torch.randn(1, start_dim)
    output = network(rand_input)
    print(output)






