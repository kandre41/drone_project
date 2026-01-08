import torch.nn as nn

class PoseControlNet(nn.Module):
    def __init__(self, num_keypoints=13, num_controls=4):
        super(PoseControlNet, self).__init__()
        
        # Input size: Keypoints * 3 (x, y, confidence)
        input_size = num_keypoints * 3 + 6 #six additional features

        self.net = nn.Sequential(
            # Layer 1: Expansion
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents overfitting to exact pixel locations
            
            # Layer 2: Feature Processing
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 3: Compression
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(32, num_controls),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: [batch_size, num_keypoints * 3]
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
