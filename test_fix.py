import sys
import os
sys.path.append(os.path.abspath('.'))

from src.preprocessing import load_record

test_path = r"C:\Users\yeezy\Desktop\mini_project_VI\data\ptbdb\patient006\s0022lre"
print(f"Testing: {test_path}")

try:
    signal = load_record(test_path)
    print(f"Success! Signal shape: {signal.shape}")
except Exception as e:
    print(f"Failed: {e}")
