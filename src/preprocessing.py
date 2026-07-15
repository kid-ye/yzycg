import numpy as np
import scipy.signal
import wfdb
import random
import os
import pandas as pd
import neurokit2 as nk

# Filter Configuration (Bandpass 0.5Hz - 40Hz)
LOW_CUT = 0.5
HIGH_CUT = 40.0
FS = 1000.0  # PTBDB Sampling Frequency
PIECE_SAMPLES = 200   # Each piece resampled to this length
NUM_PIECES = 5        # Pieces combined per segment
SEGMENT_LEN = 1000   # NPD fixed window length
MAX_PIECE_LEN = 400   # Max allowed R-R interval in samples (at 200Hz)

def bandpass_filter(data, fs=FS, order=4):
    nyquist = 0.5 * fs
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y

def _combine_pieces(pieces):
    """Resample each piece to PIECE_SAMPLES and concatenate NUM_PIECES."""
    resampled = [scipy.signal.resample(p, PIECE_SAMPLES) for p in pieces]
    return np.concatenate(resampled)  # shape: (NUM_PIECES * PIECE_SAMPLES,)

def segment_r2r(signal_1d, fs=200):
    """
    R-peak to R-peak segmentation.
    Returns list of segments, each shape (NUM_PIECES * PIECE_SAMPLES,).
    """
    _, info = nk.ecg_peaks(signal_1d, sampling_rate=fs)
    r_peaks = info['ECG_R_Peaks']
    if len(r_peaks) < NUM_PIECES + 1:
        return []

    pieces = []
    for i in range(len(r_peaks) - 1):
        piece = signal_1d[r_peaks[i]:r_peaks[i+1]]
        if 0 < len(piece) <= MAX_PIECE_LEN:
            pieces.append(piece)

    segments = []
    for i in range(len(pieces) - NUM_PIECES + 1):
        window = pieces[i:i + NUM_PIECES]
        if len(window) == NUM_PIECES:
            segments.append(_combine_pieces(window))
    return segments

def segment_p2t(signal_1d, fs=200):
    """
    P-peak to T-peak segmentation.
    Returns list of segments, each shape (NUM_PIECES * PIECE_SAMPLES,).
    """
    try:
        _, waves = nk.ecg_delineate(signal_1d, sampling_rate=fs, method='dwt')
    except Exception:
        return []

    p_peaks = np.array(waves.get('ECG_P_Peaks', []))
    t_peaks = np.array(waves.get('ECG_T_Peaks', []))

    # Filter out NaN values
    valid_p = p_peaks[~np.isnan(p_peaks)].astype(int)
    valid_t = t_peaks[~np.isnan(t_peaks)].astype(int)

    pieces = []
    for p, t in zip(valid_p, valid_t):
        if t > p:
            piece = signal_1d[p:t]
            if len(piece) > 0:
                pieces.append(piece)

    segments = []
    for i in range(len(pieces) - NUM_PIECES + 1):
        window = pieces[i:i + NUM_PIECES]
        if len(window) == NUM_PIECES:
            segments.append(_combine_pieces(window))
    return segments

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

def normalize(segment):
    """Normalize segment amplitude to [-512, 512]."""
    s_min, s_max = segment.min(), segment.max()
    if s_max == s_min:
        return np.zeros_like(segment)
    return 1024 * (segment - s_min) / (s_max - s_min) - 512

def segment_npd(signal_1d, segment_len=SEGMENT_LEN):
    """
    No Peak Detection: randomly crop a fixed-length window.
    Returns a single segment of shape (segment_len,).
    """
    length = len(signal_1d)
    if length < segment_len:
        return None
    start = random.randint(0, length - segment_len)
    return signal_1d[start:start + segment_len]

def preprocess(signal_1d, fs=200, method='npd'):
    """
    Full pipeline: bandpass filter -> segment -> normalize.
    method: 'r2r', 'p2t', or 'npd'
    Returns list of normalized segments.
    """
    filtered = bandpass_filter(signal_1d, fs=fs)

    if method == 'r2r':
        segments = segment_r2r(filtered, fs=fs)
    elif method == 'p2t':
        segments = segment_p2t(filtered, fs=fs)
    else:  # npd
        seg = segment_npd(filtered)
        segments = [seg] if seg is not None else []

    return [normalize(s) for s in segments]

def get_patient_records(base_dir):
    # Discover all records in PTBDB structure
    records = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hea"): # Header file indicates a record
                records.append(os.path.join(root, file[:-4])) # Path without extension
    return records
