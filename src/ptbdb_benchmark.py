import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks, resample
from torch.utils.data import DataLoader, Dataset

# Ensure project root is importable when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.preprocessing as pp
from src.model import ECGEncoder
from src.data_loader import ECGDataset, subject_split, DATASETS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pearson_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_c = a - torch.mean(a, dim=1, keepdim=True)
    b_c = b - torch.mean(b, dim=1, keepdim=True)
    num = torch.sum(a_c * b_c, dim=1)
    den = torch.sqrt(torch.sum(a_c ** 2, dim=1) + eps) * torch.sqrt(torch.sum(b_c ** 2, dim=1) + eps)
    return num / den


class TripletPearsonLoss(nn.Module):
    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        r_pos = pearson_similarity(anchor, positive)
        r_neg = pearson_similarity(anchor, negative)
        d_pos = 1.0 - r_pos
        d_neg = 1.0 - r_neg
        return torch.relu(d_pos - d_neg + self.margin).mean()


class SiamesePearsonContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.7):
        super().__init__()
        self.margin = margin

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # label=1 for genuine pair, label=0 for impostor pair
        r = pearson_similarity(e1, e2)
        d = 1.0 - r
        pos = label * (d ** 2)
        neg = (1.0 - label) * (torch.relu(self.margin - d) ** 2)
        return (pos + neg).mean()


@dataclass
class Split:
    train: List[str]
    val: List[str]
    test: List[str]


class SignalRepository:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.patient_records = self._discover_patient_records()
        self._cache: Dict[str, np.ndarray] = {}

    def _discover_patient_records(self) -> Dict[str, List[str]]:
        patient_records: Dict[str, List[str]] = {}
        for root, _, files in os.walk(self.data_root):
            for f in files:
                if f.endswith(".dat"):
                    rec = os.path.abspath(os.path.join(root, f[:-4]))
                    pid = os.path.basename(root)
                    patient_records.setdefault(pid, []).append(rec)
        return {k: v for k, v in patient_records.items() if v}

    def patients(self) -> List[str]:
        return sorted(self.patient_records.keys())

    def load_record_1d(self, record_path: str) -> np.ndarray:
        if record_path not in self._cache:
            sig = pp.load_record(record_path)[0]
            self._cache[record_path] = sig.astype(np.float32)
        return self._cache[record_path]

    def random_segment(self, patient_id: str, mode: str, target_len: int = 1000, fs: int = 200) -> np.ndarray:
        records = self.patient_records[patient_id]
        for _ in range(30):
            rec = random.choice(records)
            sig = self.load_record_1d(rec)
            seg = segment_signal(sig, mode=mode, target_len=target_len, fs=fs)
            if seg is None:
                continue
            filt = pp.bandpass_filter(seg, fs=fs)
            std = np.std(filt)
            if std < 1e-8:
                continue
            norm = (filt - np.mean(filt)) / (std + 1e-6)
            return norm.reshape(1, target_len).astype(np.float32)
        return np.zeros((1, target_len), dtype=np.float32)


def segment_signal(signal: np.ndarray, mode: str, target_len: int = 1000, fs: int = 200) -> np.ndarray:
    n = len(signal)
    if n < 300:
        return None

    if mode == "npd":
        if n < target_len:
            return None
        start = random.randint(0, n - target_len)
        return signal[start:start + target_len]

    peaks, _ = find_peaks(signal, distance=max(1, int(0.25 * fs)), prominence=max(0.02, np.std(signal) * 0.25))
    if len(peaks) < 3:
        if n < target_len:
            return None
        start = random.randint(0, n - target_len)
        return signal[start:start + target_len]

    if mode == "r2r":
        i = random.randint(0, len(peaks) - 2)
        a = peaks[i]
        b = peaks[i + 1]
        if b - a < 8:
            return None
        beat = signal[a:b]
        return resample(beat, target_len).astype(np.float32)

    if mode == "p2t":
        r = random.choice(peaks[1:-1])
        start = r - int(0.20 * fs)
        end = r + int(0.45 * fs)
        if start < 0 or end > n or end - start < 20:
            return None
        p2t = signal[start:end]
        return resample(p2t, target_len).astype(np.float32)

    raise ValueError(f"Unknown segmentation mode: {mode}")


class TripletDataset(Dataset):
    def __init__(self, repo: SignalRepository, patient_ids: List[str], mode: str, samples_per_epoch: int = 4000):
        self.repo = repo
        self.patient_ids = patient_ids
        self.mode = mode
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        a_id = random.choice(self.patient_ids)
        n_id = random.choice([p for p in self.patient_ids if p != a_id])
        a = self.repo.random_segment(a_id, self.mode)
        p = self.repo.random_segment(a_id, self.mode)
        n = self.repo.random_segment(n_id, self.mode)
        return torch.from_numpy(a), torch.from_numpy(p), torch.from_numpy(n)


class PairDataset(Dataset):
    def __init__(self, repo: SignalRepository, patient_ids: List[str], mode: str, samples_per_epoch: int = 4000):
        self.repo = repo
        self.patient_ids = patient_ids
        self.mode = mode
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        if random.random() < 0.5:
            pid = random.choice(self.patient_ids)
            x1 = self.repo.random_segment(pid, self.mode)
            x2 = self.repo.random_segment(pid, self.mode)
            y = np.array([1.0], dtype=np.float32)
        else:
            p1, p2 = random.sample(self.patient_ids, 2)
            x1 = self.repo.random_segment(p1, self.mode)
            x2 = self.repo.random_segment(p2, self.mode)
            y = np.array([0.0], dtype=np.float32)
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)


def split_patients(patient_ids: List[str], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Split:
    ids = patient_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return Split(train=train, val=val, test=test)


def build_encoder(device: torch.device) -> ECGEncoder:
    enc = ECGEncoder(in_channels=1, embed_dim=2304).to(device)
    return enc


def train_triplet(encoder: ECGEncoder, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    criterion = TripletPearsonLoss(margin=0.4)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    encoder.train()

    for ep in range(epochs):
        losses = []
        for a, p, n in loader:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            optimizer.zero_grad()
            ea = F.normalize(encoder(a), dim=1)
            epv = F.normalize(encoder(p), dim=1)
            env = F.normalize(encoder(n), dim=1)
            loss = criterion(ea, epv, env)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"  Epoch {ep + 1}/{epochs} - loss={np.mean(losses):.4f}")


def train_siamese(encoder: ECGEncoder, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    criterion = SiamesePearsonContrastiveLoss(margin=0.7)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    encoder.train()

    for ep in range(epochs):
        losses = []
        for x1, x2, y in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device).squeeze(1)
            optimizer.zero_grad()
            e1 = F.normalize(encoder(x1), dim=1)
            e2 = F.normalize(encoder(x2), dim=1)
            loss = criterion(e1, e2, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"  Epoch {ep + 1}/{epochs} - loss={np.mean(losses):.4f}")


def make_profiles(
    encoder: ECGEncoder,
    repo: SignalRepository,
    patient_ids: List[str],
    mode: str,
    n_enroll: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    encoder.eval()
    profiles: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid in patient_ids:
            emb_list = []
            for _ in range(n_enroll):
                seg = repo.random_segment(pid, mode)
                x = torch.from_numpy(seg).unsqueeze(0).to(device)
                emb = F.normalize(encoder(x), dim=1)
                emb_list.append(emb.squeeze(0))
            mean_emb = torch.stack(emb_list, dim=0).mean(dim=0, keepdim=True)
            mean_emb = F.normalize(mean_emb, dim=1).squeeze(0)
            profiles[pid] = mean_emb
    return profiles


def collect_scores(
    encoder: ECGEncoder,
    repo: SignalRepository,
    profiles: Dict[str, torch.Tensor],
    patient_ids: List[str],
    mode: str,
    genuine_per_user: int,
    impostor_per_user: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    scores: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for pid in patient_ids:
            # Genuine attempts
            for _ in range(genuine_per_user):
                seg = repo.random_segment(pid, mode)
                x = torch.from_numpy(seg).unsqueeze(0).to(device)
                emb = F.normalize(encoder(x), dim=1)
                ref = profiles[pid].unsqueeze(0).to(device)
                s = pearson_similarity(emb, ref).item()
                scores.append(s)
                labels.append(1)

            # Impostor attempts: another user against pid profile
            others = [u for u in patient_ids if u != pid]
            for _ in range(impostor_per_user):
                imp = random.choice(others)
                seg = repo.random_segment(imp, mode)
                x = torch.from_numpy(seg).unsqueeze(0).to(device)
                emb = F.normalize(encoder(x), dim=1)
                ref = profiles[pid].unsqueeze(0).to(device)
                s = pearson_similarity(emb, ref).item()
                scores.append(s)
                labels.append(0)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


def metrics_from_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return precision, recall, accuracy


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_t = 0.0
    best_acc = -1.0
    for t in np.linspace(-1.0, 1.0, 401):
        _, _, acc = metrics_from_threshold(scores, labels, t)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def run_experiment(
    repo: SignalRepository,
    split: Split,
    framework: str,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    encoder = build_encoder(device)

    if framework == "triplet":
        train_ds = TripletDataset(repo, split.train, mode=mode, samples_per_epoch=max(2000, len(split.train) * 80))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        train_triplet(encoder, train_loader, device=device, epochs=epochs, lr=lr)
    elif framework == "siamese":
        train_ds = PairDataset(repo, split.train, mode=mode, samples_per_epoch=max(2000, len(split.train) * 80))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        train_siamese(encoder, train_loader, device=device, epochs=epochs, lr=lr)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Threshold calibration on validation identities
    val_profiles = make_profiles(encoder, repo, split.val, mode=mode, n_enroll=5, device=device)
    val_scores, val_labels = collect_scores(
        encoder,
        repo,
        val_profiles,
        split.val,
        mode=mode,
        genuine_per_user=8,
        impostor_per_user=8,
        device=device,
    )
    threshold = tune_threshold(val_scores, val_labels)

    # Final test metrics
    test_profiles = make_profiles(encoder, repo, split.test, mode=mode, n_enroll=5, device=device)
    test_scores, test_labels = collect_scores(
        encoder,
        repo,
        test_profiles,
        split.test,
        mode=mode,
        genuine_per_user=12,
        impostor_per_user=12,
        device=device,
    )
    precision, recall, accuracy = metrics_from_threshold(test_scores, test_labels, threshold)
    return precision * 100.0, recall * 100.0, accuracy * 100.0


def print_table(rows: List[Tuple[str, str, float, float, float]]) -> None:
    print("\nPTBDB Benchmark Results")
    print("Segmentation | Framework | Precision (%) | Recall (%) | Accuracy (%)")
    print("-" * 72)
    for seg, fw, p, r, a in rows:
        print(f"{seg.upper():<12} | {fw.capitalize():<9} | {p:>12.2f} | {r:>10.2f} | {a:>11.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG benchmark across PTBDB / MITDB / ECGIDDB")
    parser.add_argument("--ptbdb",   type=str, default="data/ptbdb",   help="Path to PTBDB (train+test)")
    parser.add_argument("--mitdb",   type=str, default="",             help="Path to MITDB (test-only)")
    parser.add_argument("--ecgiddb", type=str, default="",             help="Path to ECGIDDB (test-only)")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--frameworks",   type=str,   default="siamese,triplet")
    parser.add_argument("--segmentations",type=str,   default="npd,r2r,p2t")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frameworks   = [x.strip().lower() for x in args.frameworks.split(",")    if x.strip()]
    segmentations= [x.strip().lower() for x in args.segmentations.split(",") if x.strip()]

    # -----------------------------------------------------------------------
    # PTBDB — train + test (Section VI-A: all records used for training)
    # -----------------------------------------------------------------------
    if not os.path.exists(args.ptbdb):
        raise FileNotFoundError(f"PTBDB not found: {args.ptbdb}")

    results: List[Tuple[str, str, str, float, float, float]] = []

    for seg in segmentations:
        for fw in frameworks:
            print(f"\n=== PTBDB | seg={seg.upper()} | framework={fw} ===")
            repo = SignalRepository(args.ptbdb)
            split = split_patients(repo.patients(), seed=args.seed)
            p, r, a = run_experiment(repo, split, fw, seg,
                                     args.epochs, args.batch_size, args.lr, device)
            results.append(('PTBDB', seg, fw, p, r, a))
            print(f"  Precision={p:.2f}  Recall={r:.2f}  Accuracy={a:.2f}")

    # -----------------------------------------------------------------------
    # MITDB — test-only with model trained on PTBDB (unseen dataset)
    # -----------------------------------------------------------------------
    if args.mitdb and os.path.exists(args.mitdb):
        print("\n=== MITDB (unseen, test-only) ===")
        mitdb_ds = ECGDataset(args.mitdb, dataset='mitdb')
        print(f"  Subjects: {mitdb_ds.subject_count()}  Records: {mitdb_ds.record_count()}")
        # Use last trained encoder (PTBDB, last seg/fw combo) for cross-dataset test
        # For a full sweep, loop over seg/fw as above
        results.append(('MITDB', '-', '-', 0.0, 0.0, 0.0))  # placeholder
        print("  (Run cross-dataset evaluation with --mitdb flag and a saved model)")

    # -----------------------------------------------------------------------
    # ECGIDDB — test-only (unseen dataset)
    # -----------------------------------------------------------------------
    if args.ecgiddb and os.path.exists(args.ecgiddb):
        print("\n=== ECGIDDB (unseen, test-only) ===")
        ecgid_ds = ECGDataset(args.ecgiddb, dataset='ecgiddb')
        print(f"  Subjects: {ecgid_ds.subject_count()}  Records: {ecgid_ds.record_count()}")
        results.append(('ECGIDDB', '-', '-', 0.0, 0.0, 0.0))  # placeholder
        print("  (Run cross-dataset evaluation with --ecgiddb flag and a saved model)")

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print("\nBenchmark Results")
    print(f"{'Dataset':<10} {'Seg':<5} {'Framework':<10} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-" * 58)
    for dataset, seg, fw, p, r, a in results:
        print(f"{dataset:<10} {seg.upper():<5} {fw.capitalize():<10} {p:>9.2f}% {r:>7.2f}% {a:>9.2f}%")


if __name__ == "__main__":
    main()
