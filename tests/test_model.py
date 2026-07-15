import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import TripletECGModel

# Test if model produces consistent embeddings
model = TripletECGModel()
model.eval()

# Create two identical inputs
test_input = torch.randn(1, 1, 1000)

with torch.no_grad():
    emb1 = model.encoder(test_input)
    emb2 = model.encoder(test_input)
    
    # Should be identical
    diff = (emb1 - emb2).abs().max().item()
    print(f"Same input difference: {diff}")
    
    # Create different input
    test_input2 = torch.randn(1, 1, 1000)
    emb3 = model.encoder(test_input2)
    
    # Calculate correlation
    def pearson(x, y):
        vx = x - x.mean()
        vy = y - y.mean()
        return (vx @ vy / (vx.norm() * vy.norm() + 1e-8)).item()
    
    same_corr = pearson(emb1.squeeze(), emb2.squeeze())
    diff_corr = pearson(emb1.squeeze(), emb3.squeeze())
    
    print(f"Same input correlation: {same_corr:.4f} (should be 1.0)")
    print(f"Different input correlation: {diff_corr:.4f}")
    print(f"\nEmbedding stats:")
    print(f"  Mean: {emb1.mean().item():.4f}")
    print(f"  Std: {emb1.std().item():.4f}")
    print(f"  Min: {emb1.min().item():.4f}")
    print(f"  Max: {emb1.max().item():.4f}")

# Load the actual model
try:
    state_dict = torch.load("models/ecg_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print("\n✓ Model weights loaded (with warnings)")
    
    with torch.no_grad():
        emb_loaded = model.encoder(test_input)
        print(f"\nLoaded model embedding stats:")
        print(f"  Mean: {emb_loaded.mean().item():.4f}")
        print(f"  Std: {emb_loaded.std().item():.4f}")
        
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
