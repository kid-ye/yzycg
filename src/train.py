import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import sys

class PearsonContrastiveLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(PearsonContrastiveLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-8

    def forward(self, anchor, positive, negative):
        # Pearson Correlation score:
        # r(X, Y) = sum((X-meanX)(Y-meanY)) / (sqrt(sum(X-meanX)^2) * sqrt(sum(Y-meanY)^2))
        
        # Calculate means
        mean_a = torch.mean(anchor, dim=1, keepdim=True)
        mean_p = torch.mean(positive, dim=1, keepdim=True)
        mean_n = torch.mean(negative, dim=1, keepdim=True)
        
        # Centered vectors
        a_c = anchor - mean_a
        p_c = positive - mean_p
        n_c = negative - mean_n
        
        # Calculate correlation (Similarity)
        # r_pos = Correlation(Anchor, Positive)
        num_pos = torch.sum(a_c * p_c, dim=1)
        den_pos = torch.sqrt(torch.sum(a_c ** 2, dim=1) + self.epsilon) * torch.sqrt(torch.sum(p_c ** 2, dim=1) + self.epsilon)
        r_pos = num_pos / den_pos # Value between -1 and 1
        
        # r_neg = Correlation(Anchor, Negative)
        num_neg = torch.sum(a_c * n_c, dim=1)
        den_neg = torch.sqrt(torch.sum(a_c ** 2, dim=1) + self.epsilon) * torch.sqrt(torch.sum(n_c ** 2, dim=1) + self.epsilon)
        r_neg = num_neg / den_neg # Value between -1 and 1

        # We want r_pos to be HIGH (near 1) and r_neg to be LOW (near 0 or -1).
        # Distance = 1 - r_pos. (We want minimal distance).
        # Distance_Neg = 1 - r_neg. (We want maximal distance).
        
        d_pos = 1 - r_pos
        d_neg = 1 - r_neg
        
        # Triplet Loss: max(0, d_pos - d_neg + margin)
        # If d_pos is small (0) and d_neg is big (1), then 0 - 1 + 0.5 = -0.5 -> 0 Loss! (Good)
        # If d_pos is big (1) and d_neg is small (0), then 1 - 0 + 0.5 = 1.5 Loss! (Bad)
        
        losses = torch.relu(d_pos - d_neg + self.margin)
        return torch.mean(losses)

def train(model, loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PearsonContrastiveLoss(margin=0.5) # Lambda=0.5
    
    device = next(model.parameters()).device # Use model's device
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
        
        for i, (anc, pos, neg) in pbar:
            # Move to device
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            
            optimizer.zero_grad()
            
            emb_a = model.encoder(anc) # 1D CNN forward
            emb_p = model.encoder(pos)
            emb_n = model.encoder(neg)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i+1)})

    print("Training Complete")
    torch.save(model.state_dict(), "ecg_model.pth")

if __name__ == "__main__":
    # Ensure project root is in path so 'src' module can be imported
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from src.dataset import get_loader
    from src.model import TripletECGModel
    
    # Path to data (includes ptbdb and users folders)
    DATA_PATH = "data"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} not found. Please run download_data.py first.")
        sys.exit(1)
        
    loader = get_loader(DATA_PATH, batch_size=32)
    model = TripletECGModel()
    
    # Check if GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Increase Epochs and LR since we have very small data for test
    train(model, loader, epochs=50, lr=0.0001)
