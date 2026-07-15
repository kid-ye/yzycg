"""
run_all.py  —  Train & evaluate all 6 combinations from Table I, then
               print a side-by-side comparison against the paper's numbers.

Usage:
    python run_all.py
    python run_all.py --epochs 30 --batch 64 --data data/ptbdb

Outputs (saved to results/ folder):
    results/table1_results.csv      — all metrics as CSV
    results/table1_results.txt      — formatted comparison table
    ecg_model_<seg>_<fw>.pth        — trained model per combo
"""
import argparse
import os
import sys
import time
import csv
from datetime import datetime

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ptbdb_benchmark import (
    SignalRepository,
    split_patients,
    run_experiment,
    set_seed,
    build_encoder,
    train_triplet,
    train_siamese,
    TripletDataset,
    PairDataset,
    make_profiles,
    collect_scores,
    tune_threshold,
    metrics_from_threshold,
)
from torch.utils.data import DataLoader

# ── Paper reference numbers (Table I) ────────────────────────────────────────
PAPER = {
    ("npd", "siamese"): (96.33, 98.70, 97.53),
    ("npd", "triplet"): (98.52, 99.78, 99.15),
    ("r2r", "siamese"): (95.81, 96.37, 96.10),
    ("r2r", "triplet"): (97.06, 98.04, 97.56),
    ("p2t", "siamese"): (98.04, 97.67, 97.85),
    ("p2t", "triplet"): (99.28, 99.04, 99.16),
}

COMBOS = [
    ("npd", "siamese"),
    ("npd", "triplet"),
    ("r2r", "siamese"),
    ("r2r", "triplet"),
    ("p2t", "siamese"),
    ("p2t", "triplet"),
]


def save_results(results: dict, args, elapsed_total: float, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "table1_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Segmentation", "Framework",
                         "Prec_ours", "Prec_paper",
                         "Rec_ours",  "Rec_paper",
                         "Acc_ours",  "Acc_paper"])
        for seg, fw in COMBOS:
            if (seg, fw) not in results:
                continue
            p, r, a = results[(seg, fw)]
            pp, pr, pa = PAPER[(seg, fw)]
            writer.writerow([seg.upper(), fw.capitalize(),
                             f"{p:.2f}", f"{pp:.2f}",
                             f"{r:.2f}", f"{pr:.2f}",
                             f"{a:.2f}", f"{pa:.2f}"])
    print(f"CSV saved  → {csv_path}")

    # ── Text report ──────────────────────────────────────────────────────
    txt_path = os.path.join(out_dir, "table1_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Run timestamp : {timestamp}\n")
        f.write(f"Epochs        : {args.epochs}\n")
        f.write(f"Batch size    : {args.batch}\n")
        f.write(f"Learning rate : {args.lr}\n")
        f.write(f"Data path     : {args.data}\n")
        f.write(f"Total time    : {elapsed_total/60:.1f} min\n")
        f.write("\n")

        header = (f"{'Seg':<5}  {'Framework':<9}  "
                  f"{'Prec(ours)':>10}  {'Prec(paper)':>11}  "
                  f"{'Rec(ours)':>9}  {'Rec(paper)':>10}  "
                  f"{'Acc(ours)':>9}  {'Acc(paper)':>10}")
        sep = "-" * len(header)
        f.write("=" * len(header) + "\n")
        f.write("TABLE I  —  Our results vs. Paper\n")
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for seg, fw in COMBOS:
            if (seg, fw) not in results:
                continue
            p, r, a = results[(seg, fw)]
            pp, pr, pa = PAPER[(seg, fw)]
            line = (f"{seg.upper():<5}  {fw.capitalize():<9}  "
                    f"{p:>10.2f}  {pp:>11.2f}  "
                    f"{r:>9.2f}  {pr:>10.2f}  "
                    f"{a:>9.2f}  {pa:>10.2f}")
            f.write(line + "\n")
        f.write("=" * len(header) + "\n")
    print(f"Report saved → {txt_path}")


def run_and_save_combo(repo, split, seg, fw, args, device, out_dir="results"):
    """Train, evaluate, save model, return (p, r, a)."""
    encoder = build_encoder(device)

    samples = max(2000, len(split.train) * 80)
    if fw == "triplet":
        ds = TripletDataset(repo, split.train, mode=seg, samples_per_epoch=samples)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
        train_triplet(encoder, loader, device=device, epochs=args.epochs, lr=args.lr)
    else:
        ds = PairDataset(repo, split.train, mode=seg, samples_per_epoch=samples)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
        train_siamese(encoder, loader, device=device, epochs=args.epochs, lr=args.lr)

    # Save model for this combo
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"ecg_model_{seg}_{fw}.pth")
    torch.save(encoder.state_dict(), model_path)
    print(f"  Model saved → {model_path}")

    # Validate + test
    val_profiles = make_profiles(encoder, repo, split.val,
                                 mode=seg, n_enroll=5, device=device)
    val_scores, val_labels = collect_scores(
        encoder, repo, val_profiles, split.val, mode=seg,
        genuine_per_user=8, impostor_per_user=8, device=device)
    threshold = tune_threshold(val_scores, val_labels)

    test_profiles = make_profiles(encoder, repo, split.test,
                                  mode=seg, n_enroll=5, device=device)
    test_scores, test_labels = collect_scores(
        encoder, repo, test_profiles, split.test, mode=seg,
        genuine_per_user=12, impostor_per_user=12, device=device)

    # Save raw scores for this combo (useful for FAR/FRR plots later)
    scores_path = os.path.join(out_dir, f"scores_{seg}_{fw}.npz")
    import numpy as np
    np.savez(scores_path, scores=test_scores, labels=test_labels, threshold=threshold)
    print(f"  Scores saved → {scores_path}")

    p, r, a = metrics_from_threshold(test_scores, test_labels, threshold)
    return p * 100.0, r * 100.0, a * 100.0


    seg_w, fw_w = 5, 9
    col_w = 10

    header = (f"{'Seg':<{seg_w}}  {'Framework':<{fw_w}}  "
              f"{'Prec(ours)':>{col_w}}  {'Prec(paper)':>{col_w}}  "
              f"{'Rec(ours)':>{col_w}}  {'Rec(paper)':>{col_w}}  "
              f"{'Acc(ours)':>{col_w}}  {'Acc(paper)':>{col_w}}")
    print("\n" + "=" * len(header))
    print("TABLE I  —  Our results vs. Paper")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for seg, fw in COMBOS:
        key = (seg, fw)
        if key not in results:
            continue
        p, r, a = results[key]
        pp, pr, pa = PAPER[key]
        print(f"{seg.upper():<{seg_w}}  {fw.capitalize():<{fw_w}}  "
              f"{p:>{col_w}.2f}  {pp:>{col_w}.2f}  "
              f"{r:>{col_w}.2f}  {pr:>{col_w}.2f}  "
              f"{a:>{col_w}.2f}  {pa:>{col_w}.2f}")

    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/ptbdb")
    parser.add_argument("--epochs", type=int,   default=20)
    parser.add_argument("--batch",  type=int,   default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--combos", default="all",
                        help="Comma-separated e.g. 'npd-triplet,p2t-siamese' or 'all'")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: data path '{args.data}' not found.")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Data   : {args.data}")
    print(f"Epochs : {args.epochs}   Batch : {args.batch}   LR : {args.lr}\n")

    # Select which combos to run
    if args.combos == "all":
        to_run = COMBOS
    else:
        to_run = []
        for token in args.combos.split(","):
            parts = token.strip().split("-")
            if len(parts) == 2:
                to_run.append((parts[0].lower(), parts[1].lower()))

    repo  = SignalRepository(args.data)
    split = split_patients(repo.patients(), seed=args.seed)
    print(f"Subjects: total={len(repo.patients())}  "
          f"train={len(split.train)}  val={len(split.val)}  test={len(split.test)}\n")

    results = {}
    total = len(to_run)
    t_start = time.time()

    for idx, (seg, fw) in enumerate(to_run, 1):
        print(f"\n[{idx}/{total}]  seg={seg.upper()}  framework={fw}")
        t0 = time.time()

        p, r, a = run_and_save_combo(repo, split, seg, fw, args, device, out_dir="results")

        elapsed = time.time() - t0
        results[(seg, fw)] = (p, r, a)

        pp, pr, pa = PAPER[(seg, fw)]
        print(f"  →  Prec={p:.2f}% (paper {pp:.2f})  "
              f"Rec={r:.2f}% (paper {pr:.2f})  "
              f"Acc={a:.2f}% (paper {pa:.2f})  "
              f"[{elapsed/60:.1f} min]")

    elapsed_total = time.time() - t_start
    print_table(results)
    save_results(results, args, elapsed_total)


if __name__ == "__main__":
    main()
