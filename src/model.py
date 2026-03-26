import torch
import torch.nn as nn
import torch.nn.functional as F

# The paper figure/description snippet suggests:
# - 6 convolutional layers (ReLU)
# - Max Pooling layers (red blocks in Fig 4?)
# - Final vector flattened to 2304?
# - Input shape: (Batch, 12, 1000) or (Batch, 1, 1000)?
# Usually if using all 12 leads, one would use 12 input channels.
# If single lead, then 1 input channel.
# Let's assume 12 input channels for general case or 1 if specific lead.

class ECGEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_size=2304):
        super(ECGEncoder, self).__init__()
        
        # Architecture inspired by standard 1D CNN for ECG
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2) # Downsample by 2
        
        # Layer 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)

        # Layer 4
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(2)
        
        # Layer 5
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(2)
        
        # Layer 6
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(2)

        # Flatten + Linear?
        # Output shape after 6 maxpools:
        # Input 1000 -> 500 -> 250 -> 125 -> 62 -> 31 -> 15 (Approx)
        # Final shape: 256 filters * 15 length = 3840 features
        # Flatten -> Dense -> Vector (2304)
        
        self.fc = nn.Linear(3840, hidden_size)

    def forward(self, x):
        # x shape: (Batch, Channels, Length=1000)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x) # Fully connected to match paper output size
        return x

class TripletECGModel(nn.Module):
    def __init__(self):
        super(TripletECGModel, self).__init__()
        self.encoder = ECGEncoder()
        
    def forward(self, anchor, positive, negative):
        # If passed as triplet
        emb_a = self.encoder(anchor)
        emb_p = self.encoder(positive)
        emb_n = self.encoder(negative)
        return emb_a, emb_p, emb_n

    def encode(self, x):
        return self.encoder(x)
