import torch.nn as nn
import torch.nn.functional as F

class DeepAmharicNet(nn.Module):
    """
    A Deeper CNN for Amharic Character Recognition.
    
    Architecture:
    - 4 Convolutional Blocks (Increasing depth: 32 -> 64 -> 128 -> 256)
    - Batch Normalization in every block for stability.
    - Dropout in fully connected layers to reduce overfitting.
    
    Target: >65% Accuracy on 238 classes.
    """
    def __init__(self, num_classes=238):
        super(DeepAmharicNet, self).__init__()
        
        # Block 1: 32 Filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 64 Filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 128 Filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: 256 Filters
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2) # Divides size by 2
        
        # Fully Connected Layers
        # Input size calculation:
        # Start: 32x32
        # After Pool 1: 16x16
        # After Pool 2: 8x8
        # After Pool 3: 4x4
        # After Pool 4: 2x2
        # Final Feature Map: 256 channels * 2 * 2 = 1024
        
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
