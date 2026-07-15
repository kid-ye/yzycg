"""
Dataset loaders for Section VI-A:
  - PTBDB  : 549 records, 290 subjects, 1000 Hz, wfdb .dat/.hea
  - MITDB  :  48 records,  47 subjects,  360 Hz, wfdb .dat/.hea
  - ECGIDDB: 310 records,  90 subjects,  500 Hz, wfdb .dat/.hea

All signals are resampled to 200 Hz and bandpass-filtered (0.5–40 Hz)
before segmentation, matching Section III-A of the paper.
"""
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import wfdb

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.preprocessing as pp

TARGET_FS = 200   # Hz — unified sampling rate for all datasets


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS = {
    'ptbdb':   {'fs': 1000, 'channel': 0},   # Lead I
    'mitdb':   {'fs':  360, 'channel': 0},   # MLII
    'ecgiddb': {'fs':  500, 'channel': 0},   # Lead I (filtered channel)
}


# ---------------------------------------------------------------------------
# Record discovery — groups records by subject/patient folder
# ---------------------------------------------------------------------------

def find_subject_records(data_root: str) -> Dict[str, List[str]]:
    """
    Walk data_root and group wfdb record paths by their parent folder name
    (= subject/patient identity).
    Returns {subject_id: [record_path, ...]}
    """
    subjects: Dict[str, List[str]] = {}
    for dirpath, _, files in os.walk(data_root):
        for f in files:
            if f.endswith('.hea'):
                rec_path = os.path.join(dirpath, f[:-4])
                subject_id = os.path.basename(dirpath)
                subjects.setdefault(subject_id, []).append(rec_path)
    return {k: sorted(v) for k, v in subjects.items() if v}


# ---------------------------------------------------------------------------
# Signal loading with per-dataset resampling
# ---------------------------------------------------------------------------

def load_signal(record_path: str, source_fs: int, channel: int = 0) -> Optional[np.ndarray]:
    """
    Load one channel from a wfdb record, resample to TARGET_FS.
    Returns 1-D float32 array or None on failure.
    """
    try:
        rec = wfdb.rdrecord(record_path, channels=[channel])
        sig = rec.p_signal[:, 0].astype(np.float32)

        if source_fs != TARGET_FS:
            n_out = int(len(sig) * TARGET_FS / source_fs)
            sig = scipy.signal.resample(sig, n_out).astype(np.float32)

        return sig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Segment extraction (bandpass + normalize + crop)
# ---------------------------------------------------------------------------

def extract_segment(sig: np.ndarray, method: str = 'npd') -> Optional[np.ndarray]:
    """
    Apply full preprocessing pipeline and return one 1000-sample segment.
    """
    segments = pp.preprocess(sig, fs=TARGET_FS, method=method)
    return segments[0] if segments else None


# ---------------------------------------------------------------------------
# Subject-level dataset class
# ---------------------------------------------------------------------------

class ECGDataset:
    """
    Unified interface for PTBDB, MITDB, and ECGIDDB.

    Usage:
        ds = ECGDataset('data/ptbdb', dataset='ptbdb')
        seg = ds.get_segment('patient001')   # numpy (1000,)
        subjects = ds.subjects()
    """

    def __init__(self, data_root: str, dataset: str, seg_method: str = 'npd'):
        if dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {list(DATASETS.keys())}")
        self.data_root  = data_root
        self.source_fs  = DATASETS[dataset]['fs']
        self.channel    = DATASETS[dataset]['channel']
        self.seg_method = seg_method
        self.subject_records = find_subject_records(data_root)
        self._sig_cache: Dict[str, np.ndarray] = {}

    def subjects(self) -> List[str]:
        return sorted(self.subject_records.keys())

    def _load_cached(self, record_path: str) -> Optional[np.ndarray]:
        if record_path not in self._sig_cache:
            sig = load_signal(record_path, self.source_fs, self.channel)
            if sig is not None:
                self._sig_cache[record_path] = sig
        return self._sig_cache.get(record_path)

    def get_segment(self, subject_id: str, max_attempts: int = 20) -> Optional[np.ndarray]:
        """Return one preprocessed 1000-sample segment for a subject."""
        records = self.subject_records.get(subject_id, [])
        if not records:
            return None
        for _ in range(max_attempts):
            rec = random.choice(records)
            sig = self._load_cached(rec)
            if sig is None:
                continue
            seg = extract_segment(sig, self.seg_method)
            if seg is not None:
                return seg
        return None

    def get_n_segments(self, subject_id: str, n: int) -> List[np.ndarray]:
        """Return up to n segments for a subject (used for enrollment)."""
        segs = []
        for _ in range(n * 5):          # extra attempts to fill n
            seg = self.get_segment(subject_id)
            if seg is not None:
                segs.append(seg)
            if len(segs) == n:
                break
        return segs

    def subject_count(self) -> int:
        return len(self.subject_records)

    def record_count(self) -> int:
        return sum(len(v) for v in self.subject_records.values())


# ---------------------------------------------------------------------------
# Train / val / test split at subject level
# ---------------------------------------------------------------------------

def subject_split(
    subjects: List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split subjects into train / val / test.
    PTBDB is used for train+test; MITDB and ECGIDDB are test-only
    (pass all subjects as test_subjects directly in those cases).
    """
    rng = random.Random(seed)
    ids = subjects[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


# ---------------------------------------------------------------------------
# Quick summary utility
# ---------------------------------------------------------------------------

def dataset_summary(data_root: str, dataset: str) -> None:
    ds = ECGDataset(data_root, dataset)
    print(f"\n{dataset.upper()}")
    print(f"  Subjects : {ds.subject_count()}")
    print(f"  Records  : {ds.record_count()}")
    print(f"  Source FS: {ds.source_fs} Hz  →  resampled to {TARGET_FS} Hz")
    # Verify one segment loads correctly
    subj = ds.subjects()[0]
    seg = ds.get_segment(subj)
    status = f"shape={seg.shape}, range=[{seg.min():.1f}, {seg.max():.1f}]" if seg is not None else "FAILED"
    print(f"  Sample segment ({subj}): {status}")


if __name__ == '__main__':
    for name, path in [('ptbdb',   'data/ptbdb'),
                       ('mitdb',   'data/mitdb'),
                       ('ecgiddb', 'data/ecgiddb')]:
        if os.path.exists(path):
            dataset_summary(path, name)
        else:
            print(f"\n{name.upper()}: data not found at '{path}'")
