import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
import wfdb
import src.preprocessing as pp

# Custom Dataset for Triplet Sampling
class TripletECGDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        # Gather all .dat files (record paths)
        self.patient_records = self._find_records()
        self.transform = transform 

    def _find_records(self):
        records = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(".dat"):
                    # PTBDB often has multiple records per patient
                    # but we can treat each record as a source or group by patient.
                    # For simplicity, treat each *folder* as a patient identity?
                    # The paper uses "290 subjects". PTBDB has 'patientXXX'.
                    # We should group records by patient folder.
                    patient_id = os.path.basename(root)
                    record_path = os.path.abspath(os.path.join(root, file[:-4]))
                    records.append(record_path)
                elif file.endswith(".csv") or file.endswith(".xlsx"):
                    # Also include custom user data (Excel/CSV)
                    # We'll use the full filepath for these
                    record_path = os.path.join(root, file)
                    records.append(record_path)        
        # Group by patient
        patient_map = {}
        for r in records:
            pid = os.path.basename(os.path.dirname(r))
            if pid not in patient_map:
                patient_map[pid] = []
            patient_map[pid].append(r)
            
        return list(patient_map.values()) # List of Lists of record paths

    def __len__(self):
        return len(self.patient_records)

    def __getitem__(self, idx):
        # Triplet Logic:
        # Anchor (A): From patient[idx]
        # Positive (P): Different segment from patient[idx]
        # Negative (N): Segment from ANY OTHER patient[j != idx]
        
        patient_recs = self.patient_records[idx]
        
        # Load anchor signal
        anchor_signal = self._load_random_segment(patient_recs)
        
        # Load positive signal (same patient, new random segment)
        positive_signal = self._load_random_segment(patient_recs)
        
        # Load negative signal (different patient)
        neg_idx = random.choice([i for i in range(len(self.patient_records)) if i != idx])
        neg_patient_recs = self.patient_records[neg_idx]
        negative_signal = self._load_random_segment(neg_patient_recs)
        
        return torch.tensor(anchor_signal).float(), torch.tensor(positive_signal).float(), torch.tensor(negative_signal).float()

    def _load_random_segment(self, patient_records):
        # Pick a random record from the patient
        # Using preprocessing utility
        import src.preprocessing as pp
        
        max_attempts = 10
        for _ in range(max_attempts):
            rec_path = random.choice(patient_records)
            try:
                # Load: (1, Time)
                if rec_path.endswith('.csv') or rec_path.endswith('.xlsx'):
                    full_signal = pp.load_util_file(rec_path)
                else:
                    full_signal = pp.load_record(rec_path)
                
                # Check length
                if full_signal.shape[1] < 1000:
                    print(f"Skipping {rec_path}: too short ({full_signal.shape[1]})")
                    continue 

                # Crop random 1000 segment
                start = random.randint(0, full_signal.shape[1] - 1000)
                segment = full_signal[:, start:start+1000]
                
                # Filter (using preprocessing)
                filtered_segment = pp.bandpass_filter(segment[0], fs=200) # shape (1000,)
                
                # Normalize?
                std_val = np.std(filtered_segment)
                if std_val == 0:
                    print(f"Skipping {rec_path}: Flat signal (std=0)")
                    continue
                    
                segment_std = (filtered_segment - np.mean(filtered_segment)) / (std_val + 1e-6)

                return segment_std.reshape(1, 1000)
            except Exception as e:
                print(f"Error loading {rec_path}: {e}")
                continue
                
        # If failure, return zeros (bad practice but keeps it running)
        print("Warning: _load_random_segment failed after max attempts! Returning zeros.")
        return np.zeros((1, 1000))

def get_loader(root, batch_size=32):
    ds = TripletECGDataset(root)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
