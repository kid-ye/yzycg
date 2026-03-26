import wfdb
import numpy as np
import src.preprocessing as pp
import os

try:
    # Try loading a specific file we saw
    path = "data/ptbdb/patient001/s0010_re"
    if not os.path.exists(path + ".dat"):
        print(f"File not found: {path}.dat")
    else:
        print(f"Loading {path}...")
        signal = pp.load_record(path)
        print(f"Signal shape: {signal.shape}")
        print(f"Signal mean: {np.mean(signal)}")
        print(f"Signal std: {np.std(signal)}")
        print(f"Signal max: {np.max(signal)}")
        
        # Check filtered segment
        filtered = pp.bandpass_filter(signal[0], fs=200)
        print(f"Filtered mean: {np.mean(filtered)}")
        print(f"Filtered std: {np.std(filtered)}")
        
        if np.std(filtered) == 0:
            print("WARNING: Signal is flat (std=0)!")
        else:
            print("Signal looks okay (non-zero variance).")

except Exception as e:
    print(f"Error: {e}")
