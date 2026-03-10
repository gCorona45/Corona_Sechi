import torch
import torch.nn as nn

class DQN_Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_Network, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(128, action_dim)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_layer(x)
        return self.output_layer(x)