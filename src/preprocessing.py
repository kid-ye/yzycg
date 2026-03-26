import numpy as np
import scipy.signal
import wfdb
import random
import os
import pandas as pd

# Filter Configuration (Bandpass 0.5Hz - 40Hz)
LOW_CUT = 0.5
HIGH_CUT = 40.0
FS = 1000.0  # PTBDB Sampling Frequency

def bandpass_filter(data, fs=FS, order=4):
    nyquist = 0.5 * fs
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y

def load_record(record_path):
    # Normalize path for Windows
    record_path = os.path.normpath(record_path)
    
    # Check if header file exists
    if not os.path.exists(record_path + '.hea'):
        raise FileNotFoundError(f"Header file not found: {record_path}.hea")
    
    try:
        # Read only channels that exist (skip .xyz files)
        # PTBDB has 12 standard leads in .dat, 3 XYZ leads in .xyz (often missing)
        record = wfdb.rdrecord(record_path, channels=[0])  # Read only first channel (lead I)
    except Exception as e:
        raise ValueError(f"Failed to read record {record_path}: {e}")
    
    # Extract only Lead I (Index 0 usually in PTBDB, check leads)
    # Most PTBDB records have 15 leads, first one is usually 'i' or 'I'.
    # We already read only channel 0, so just extract it
    signal = record.p_signal[:, 0] # Shape: (Time,)
    
    # Resample from 1000Hz (PTBDB) to 200Hz (Target)
    # Calculate number of samples
    original_len = len(signal)
    target_len = int(original_len * (200 / 1000))
    resampled_signal = scipy.signal.resample(signal, target_len)
    
    # Add channel dimension: (1, Time)
    return resampled_signal.reshape(1, -1)

def load_util_file(file_path, target_fs=200):
    """
    Load ECG data from CSV or Excel file.
    Assumes data is in the first numeric column.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else: # Excel
            df = pd.read_excel(file_path)
            
        # Select first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric data found")
            
        signal = df[numeric_cols[0]].values
        
        # Simple Resampling if needed (Assuming source is roughly handled or specific)
        # If we assume MAX30003 is configured for ~128Hz or 512Hz, we might want to resample.
        # For now, let's treat it as is or assume user sets it close to target.
        # But to be safe against length mismatches in model, we just ensure shape (1, T)
        
        return signal.reshape(1, -1)
        
    except Exception as e:
        print(f"Error reading custom file {file_path}: {e}")
        raise e

def segment_npd(signal, segment_len=1000):
    # No Peak Detection (NPD) segmentation: Random crop
    # signal shape: (Channels, Time)
    channels, length = signal.shape
    if length < segment_len:
        return None
    start = random.randint(0, length - segment_len)
    return signal[:, start:start+segment_len]

def get_patient_records(base_dir):
    # Discover all records in PTBDB structure
    records = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hea"): # Header file indicates a record
                records.append(os.path.join(root, file[:-4])) # Path without extension
    return records
