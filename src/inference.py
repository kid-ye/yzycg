import torch
import numpy as np
import os
import sys
import random

# Ensure project root is in path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.preprocessing as pp
from src.model import TripletECGModel

class ECGAuthenticator:
    def __init__(self, model_path="models/ecg_model.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model architecture
        self.model = TripletECGModel().to(self.device)
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print(f"Warning: Model path {model_path} not found. Using untrained weights.")
        
        self.model.eval()
        self.enrolled_embeddings = {} # {user_id: mean_embedding_tensor}

        # Auto-load existing profiles from disk if available
        self.load_profiles()

    def load_from_data_folder(self, data_path="data"):
        """
        Scans data/ptbdb/patientXXX or data/users/NAME folders.
        """
        if not os.path.exists(data_path): return
            
        print(f"Scanning {data_path} for user data...")
        count_enrolled = 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(data_path):
            has_data = any(f.endswith('.dat') or f.endswith('.csv') or f.endswith('.xlsx') for f in files)
            
            if has_data:
                user_id = os.path.basename(root)
                if user_id in self.enrolled_embeddings: continue
                
                # --- CALL HELPER FUNCTION ---
                # Ensure get_random_segments is accessible
                samples = get_random_segments(root, count=5)
                if samples:
                    print(f"  Auto-enrolling found user: {user_id}")
                    if self.enroll_user(user_id, samples):
                        count_enrolled += 1
                        
        if count_enrolled > 0:
            print(f"Auto-enrolled {count_enrolled} users.")
            self.save_profiles()

    def save_profiles(self, path="models/enrolled_profiles.pt"):
        try:
            torch.save(self.enrolled_embeddings, path)
            print(f"Profiles saved to {path} ({len(self.enrolled_embeddings)} users).")
            return True
        except Exception as e:
            print(f"Error saving profiles: {e}")
            return False

    def load_profiles(self, path="models/enrolled_profiles.pt"):
        if os.path.exists(path):
            try:
                loaded = torch.load(path, map_location=self.device)
                if isinstance(loaded, dict):
                    self.enrolled_embeddings.update(loaded)
                    print(f"Loaded {len(loaded)} profiles from {path}.")
                    return True
            except Exception as e:
                print(f"Error loading profiles: {e}")
                return False
        else:
            print(f"No existing profiles found at {path}.")
            return False

    def preprocess_signal(self, raw_signal):
        """
        Bandpass filter + normalize to [-512, 512] (paper Section II-A).
        Returns Tensor (1, 1, 1000) on device, or None if signal too short.
        """
        sig_flat = np.array(raw_signal).flatten()
        if len(sig_flat) < 1000:
            return None

        segments = pp.preprocess(sig_flat, fs=200, method='npd')
        if not segments:
            return None

        seg = segments[0]  # shape (1000,)
        tensor_sig = torch.tensor(seg, dtype=torch.float32).reshape(1, 1, 1000)
        return tensor_sig.to(self.device)

    def get_embedding(self, signal_tensor):
        with torch.no_grad():
            # Use the encoder part of TripletECGModel
            embedding = self.model.encoder(signal_tensor) # (1, 2304)
        return embedding

    def enroll_user(self, user_id, signal_list):
        """
        Enroll a user: store all embedding vectors in the database.
        self.enrolled_embeddings[user_id] = list of (1, 2304) tensors.
        """
        embeddings = []
        for sig in signal_list:
            tensor_sig = self.preprocess_signal(sig)
            if tensor_sig is not None:
                emb = self.get_embedding(tensor_sig)  # (1, 2304)
                embeddings.append(emb)

        if not embeddings:
            print(f"Failed to enroll {user_id}: No valid signals.")
            return False

        self.enrolled_embeddings[user_id] = embeddings
        print(f"User '{user_id}' enrolled with {len(embeddings)} samples.")
        return True

    @staticmethod
    def _pearson(x, y):
        """Pearson Correlation Coefficient between two 1-D tensors (Eqn 1)."""
        vx = x - x.mean()
        vy = y - y.mean()
        return (vx @ vy / (vx.norm() * vy.norm() + 1e-8)).item()

    def authenticate(self, user_id, raw_signal, threshold=0.70):
        """
        Compare live signal against all stored embeddings for user_id.
        Mapping score D(X,Y) = mean Pearson correlation across enrolled vectors.
        Returns: (is_authenticated (bool), mapping_score (float))
        """
        if user_id not in self.enrolled_embeddings:
            print(f"User '{user_id}' not enrolled.")
            return False, 0.0

        tensor_sig = self.preprocess_signal(raw_signal)
        if tensor_sig is None:
            return False, 0.0

        live_emb = self.get_embedding(tensor_sig).squeeze()   # (2304,)
        ref_embeddings = self.enrolled_embeddings[user_id]    # list of (1, 2304)

        scores = [self._pearson(live_emb, ref.squeeze()) for ref in ref_embeddings]
        mapping_score = float(np.mean(scores))

        return mapping_score > threshold, mapping_score

# --- Helper to load test data ---
def get_random_segments(patient_folder, count=5):
    segments = []
    
    if not os.path.exists(patient_folder):
        return []
        
    files = [f for f in os.listdir(patient_folder) if f.endswith('.dat') or f.endswith('.csv') or f.endswith('.xlsx')]
    if not files: 
        return []
    
    attempts = 0
    # Try getting 'count' valid segments
    while len(segments) < count and attempts < 50:
        attempts += 1
        f = random.choice(files)
        
        try:
            if f.endswith('.dat'):
                path = os.path.join(patient_folder, f[:-4]) # Remove .dat for wfdb
                full_sig = pp.load_record(path)
            else:
                 # CSV/Excel
                 path = os.path.join(patient_folder, f)
                 full_sig = pp.load_util_file(path)
            
            # Check length to ensure we can crop 1000
            if full_sig.shape[1] < 1000: 
                continue
            
            # Random crop
            start = random.randint(0, full_sig.shape[1] - 1000)
            # Take the first channel (row 0), slice [start : start+1000]
            seg = full_sig[0, start:start+1000] # Shape (1000,)
            segments.append(seg)
        except Exception as e:
            # print(f"Debug: load failed for {path}: {e}")
            continue
            
    return segments

if __name__ == "__main__":
    # Path setup
    DATA_PATH = "data"
    if not os.path.exists(DATA_PATH):
        print(f"Data directory {DATA_PATH} not found.")
        sys.exit(1)
        
    auth = ECGAuthenticator()
    
    # 1. Discover all patient folders recursively
    patient_folders = []
    for root, dirs, files in os.walk(DATA_PATH):
        # A folder is a "patient" if it contains data files
        if any(f.endswith('.dat') or f.endswith('.csv') or f.endswith('.xlsx') for f in files):
            patient_folders.append(root)
            
    if len(patient_folders) < 2:
        print(f"Need at least 2 patient folders with data in '{DATA_PATH}' to demonstrate authentication.")
        print("Folder structure example: data/ptbdb/patient001/ OR data/users/yash/")
        sys.exit(0)
        
    # Shuffle to get random pair
    random.shuffle(patient_folders)
    owner_path = patient_folders[0]
    attacker_path = patient_folders[1]
    
    owner_id = os.path.basename(owner_path)
    attacker_id = os.path.basename(attacker_path)
    
    print(f"\n--- Scenario Setup ---")
    print(f"Owner: {owner_id} ({owner_path})")
    print(f"Attacker: {attacker_id} ({attacker_path})")
    
    # 2. Enrollment Phase (Owner)
    print(f"\n[ENROLLMENT] Getting 5 samples from Owner ({owner_id})...")
    # owner_path is already full path
    enroll_samples = get_random_segments(owner_path, count=5)
    
    if len(enroll_samples) < 1:
        print("Could not get samples for enrollment.")
        sys.exit(1)
        
    auth.enroll_user("TheBoss", enroll_samples)
    
    # 3. Verification Phase (Owner vs Self)
    print(f"\n[TEST 1] Owner verifying against own profile...")
    test_samples_owner = get_random_segments(owner_path, count=5)
    
    scores_self = []
    threshold = 0.70
    
    for i, sig in enumerate(test_samples_owner):
        is_match, score = auth.authenticate("TheBoss", sig, threshold=threshold)
        scores_self.append(score)
        result = "✅ MATCH" if is_match else "❌ NO MATCH"
        print(f"  Sample {i+1}: Correlation = {score:.4f} -> {result}")
        
    avg_self = sum(scores_self)/len(scores_self) if scores_self else 0.0
    print(f"  > Average Self-Correlation: {avg_self:.4f}")

    # 4. Attack Phase (Attacker vs Owner Profile)
    print(f"\n[TEST 2] Attacker ({attacker_id}) trying to spoof Owner...")
    # attacker_path is already full path
    test_samples_attacker = get_random_segments(attacker_path, count=5)
    
    scores_diff = []
    
    for i, sig in enumerate(test_samples_attacker):
        is_match, score = auth.authenticate("TheBoss", sig, threshold=threshold)
        scores_diff.append(score)
        result = "🚨 FALSE POSITIVE" if is_match else "🛡️ BLOCKED"
        print(f"  Sample {i+1}: Correlation = {score:.4f} -> {result}")

    avg_diff = sum(scores_diff)/len(scores_diff) if scores_diff else 0.0
    print(f"  > Average Cross-Correlation: {avg_diff:.4f}")
    
    print("\n--- Conclusion ---")
    if avg_self > threshold and avg_diff < threshold:
        print("SUCCESS: System correctly authenticates owner and rejects attacker.")
    else:
        print("WARNING: System performance may need tuning (adjust threshold or train longer).")
