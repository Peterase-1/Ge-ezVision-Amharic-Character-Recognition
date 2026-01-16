import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A Simple Convolutional Neural Network (CNN) for Character Recognition.
    
    Architecture:
    1. Input Layer: (1, 32, 32) image
    2. Conv Layer 1: Finds simple features like edges.
    3. Pooling: Reduces size (32x32 -> 16x16).
    4. Conv Layer 2: Finds shapes using the edges.
    5. Pooling: Reduces size (16x16 -> 8x8).
    6. Flatten: Turns the 2D box of numbers into a long list.
    7. Fully Connected Layer: Combines features.
    8. Output Layer: Decides which of the 238 characters it is.
    """
    def __init__(self, num_classes=238):
        super(SimpleCNN, self).__init__()
        
        # Block 1
        # In: 1 channel (grayscale), Out: 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Divides size by 2
        
        # Block 2
        # In: 32 filters, Out: 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Classification Head
        # After 2 pools (32 -> 16 -> 8), the image is 64 channels x 8 x 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting (memorizing)
        self.fc2 = nn.Linear(128, num_classes) # Final output

    def forward(self, x):
        # Pass through Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) # Activation (adds non-linearity)
        x = self.pool(x)
        
        # Pass through Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten: (Batch, 64, 8, 8) -> (Batch, 4096)
        x = x.view(-1, 64 * 8 * 8)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
