import wfdb
import os

# Test different path formats
test_path = r"C:\Users\yeezy\Desktop\mini_project_VI\data\ptbdb\patient006\s0022lre"

print(f"Testing path: {test_path}")
print(f"Header exists: {os.path.exists(test_path + '.hea')}")
print(f"Data exists: {os.path.exists(test_path + '.dat')}")

# Try reading with different path formats
try:
    # Method 1: Direct path with backslashes
    print("\nMethod 1: Direct backslash path")
    record = wfdb.rdrecord(test_path)
    print(f"Success! Signal shape: {record.p_signal.shape}")
except Exception as e:
    print(f"Failed: {e}")

try:
    # Method 2: Forward slashes
    print("\nMethod 2: Forward slash path")
    test_path_forward = test_path.replace("\\", "/")
    record = wfdb.rdrecord(test_path_forward)
    print(f"Success! Signal shape: {record.p_signal.shape}")
except Exception as e:
    print(f"Failed: {e}")

try:
    # Method 3: Using pn_dir parameter
    print("\nMethod 3: Using pn_dir parameter")
    record = wfdb.rdrecord('s0022lre', pn_dir=r'C:\Users\yeezy\Desktop\mini_project_VI\data\ptbdb\patient006')
    print(f"Success! Signal shape: {record.p_signal.shape}")
except Exception as e:
    print(f"Failed: {e}")
