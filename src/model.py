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
    """
    1D-CNN Encoder from Fig. 4.
    Input : (B, 1, 1000)
    Output: (B, 2304)

    Layer dimensions (filters x kernel):
      Conv1: 16 x 5  -> Pool -> 500
      Conv2: 32 x 5  -> Pool -> 250
      Conv3: 64 x 5  -> Pool -> 125
      Conv4: 128 x 5 -> Pool ->  62
      Conv5: 256 x 3 -> Pool ->  31
      Conv6: 256 x 3 -> Pool ->  15
      Flatten: 256 * 15 = 3840
      FC: 3840 -> 2304
    """
    def __init__(self, in_channels=1, embed_dim=2304):
        super(ECGEncoder, self).__init__()

        def conv_block(in_ch, out_ch, kernel):
            padding = kernel // 2
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels,  16, 5),   # -> (B, 16, 500)
            conv_block(16,           32, 5),   # -> (B, 32, 250)
            conv_block(32,           64, 5),   # -> (B, 64, 125)
            conv_block(64,          128, 5),   # -> (B, 128, 62)
            conv_block(128,         256, 3),   # -> (B, 256, 31)
            conv_block(256,         256, 3),   # -> (B, 256, 15)
        )
        self.fc = nn.Linear(256 * 15, embed_dim)  # 3840 -> 2304

    def forward(self, x):
        # x: (B, 1, 1000)
        x = self.encoder(x)          # (B, 256, 15)
        x = x.flatten(1)             # (B, 3840)
        return self.fc(x)            # (B, 2304)

class TripletECGModel(nn.Module):
    def __init__(self):
        super(TripletECGModel, self).__init__()
        self.encoder = ECGEncoder(in_channels=1, embed_dim=2304)
        
    def forward(self, anchor, positive, negative):
        # If passed as triplet
        emb_a = self.encoder(anchor)
        emb_p = self.encoder(positive)
        emb_n = self.encoder(negative)
        return emb_a, emb_p, emb_n

    def encode(self, x):
        return self.encoder(x)
