"""
plot_far_frr.py  —  Reproduce Fig. 8: FAR/FRR plots for λ = 0.5 … 1.0

The model is trained ONCE (NPD + Triplet, best combo from Table I).
Genuine and impostor Pearson scores are collected on the test split.
For each λ value, the decision threshold IS λ itself (paper Section IV-A),
and we sweep thresholds across [-1, 1] to draw the full FAR/FRR curve,
marking the operating point at threshold = λ.

Usage:
    python plot_far_frr.py --data data/ptbdb --epochs 20 --batch 64
    python plot_far_frr.py --data data/ptbdb --load-scores scores.npz   # skip training
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless — saves PNG without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ptbdb_benchmark import (
    SignalRepository, Split, split_patients, set_seed,
    build_encoder, train_triplet, make_profiles, collect_scores,
    TripletDataset, pearson_similarity,
)

LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLDS = np.linspace(-1.0, 1.0, 1000)


# ── FAR / FRR at a single threshold ──────────────────────────────────────────

def far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    far = fp / (fp + tn + 1e-10)   # impostor acceptance rate
    frr = fn / (fn + tp + 1e-10)   # genuine rejection rate
    return far, frr


# ── Sweep thresholds → full FAR/FRR curves ───────────────────────────────────

def compute_curves(scores: np.ndarray, labels: np.ndarray):
    fars, frrs = [], []
    for t in THRESHOLDS:
        f, r = far_frr(scores, labels, t)
        fars.append(f)
        frrs.append(r)
    return np.array(fars), np.array(frrs)


# ── EER (where FAR ≈ FRR) ────────────────────────────────────────────────────

def eer(fars: np.ndarray, frrs: np.ndarray):
    diff = np.abs(fars - frrs)
    idx  = np.argmin(diff)
    return THRESHOLDS[idx], (fars[idx] + frrs[idx]) / 2


# ── Plot 6 subplots (Fig. 8) ─────────────────────────────────────────────────

def plot_fig8(scores: np.ndarray, labels: np.ndarray, out_path: str = "fig8_far_frr.png"):
    fars, frrs = compute_curves(scores, labels)
    eer_t, eer_val = eer(fars, frrs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fig. 8: FAR and FRR for varying λ", fontsize=14, fontweight="bold")

    for ax, lam in zip(axes.flat, LAMBDAS):
        far_op, frr_op = far_frr(scores, labels, threshold=lam)

        ax.plot(THRESHOLDS, fars  * 100, color="tab:red",  linewidth=1.8, label="FAR")
        ax.plot(THRESHOLDS, frrs  * 100, color="tab:blue", linewidth=1.8, label="FRR")

        # Mark operating point at threshold = λ
        ax.axvline(x=lam, color="black", linestyle="--", linewidth=1.2, label=f"λ={lam}")
        ax.plot(lam, far_op * 100, "rv", markersize=8)
        ax.plot(lam, frr_op * 100, "b^", markersize=8)

        # Mark EER
        ax.axvline(x=eer_t, color="gray", linestyle=":", linewidth=1.0)
        ax.plot(eer_t, eer_val * 100, "ko", markersize=6,
                label=f"EER={eer_val*100:.2f}%")

        ax.set_title(f"λ = {lam}", fontsize=11)
        ax.set_xlabel("Threshold", fontsize=9)
        ax.set_ylabel("Rate (%)", fontsize=9)
        ax.set_xlim(-0.1, 1.05)
        ax.set_ylim(-1, 101)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate operating point values
        ax.annotate(f"FAR={far_op*100:.1f}%", xy=(lam, far_op*100),
                    xytext=(lam + 0.05, far_op*100 + 5),
                    fontsize=7, color="tab:red",
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
        ax.annotate(f"FRR={frr_op*100:.1f}%", xy=(lam, frr_op*100),
                    xytext=(lam + 0.05, frr_op*100 - 8),
                    fontsize=7, color="tab:blue",
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFig. 8 saved → {out_path}")

    # Also print a summary table
    print(f"\n{'λ':>5}  {'FAR (%)':>8}  {'FRR (%)':>8}  {'EER (%)':>8}")
    print("-" * 38)
    for lam in LAMBDAS:
        f, r = far_frr(scores, labels, lam)
        print(f"{lam:>5.1f}  {f*100:>8.2f}  {r*100:>8.2f}  {eer_val*100:>8.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/ptbdb")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--seg",         default="npd",     help="npd | r2r | p2t")
    parser.add_argument("--framework",   default="triplet", help="triplet | siamese")
    parser.add_argument("--load-scores", default="",        help="Path to .npz to skip training")
    parser.add_argument("--save-scores", default="scores.npz")
    parser.add_argument("--out",         default="fig8_far_frr.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load pre-computed scores or train + collect ───────────────────────
    if args.load_scores and os.path.exists(args.load_scores):
        print(f"Loading scores from {args.load_scores}")
        data = np.load(args.load_scores)
        scores, labels = data["scores"], data["labels"]
    else:
        if not os.path.exists(args.data):
            print(f"ERROR: data path '{args.data}' not found.")
            sys.exit(1)

        repo  = SignalRepository(args.data)
        split = split_patients(repo.patients(), seed=args.seed)
        print(f"Subjects: train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")

        # Train
        encoder = build_encoder(device)
        print(f"\nTraining ({args.framework}, {args.seg.upper()}, {args.epochs} epochs)...")
        if args.framework == "triplet":
            ds = TripletDataset(repo, split.train, mode=args.seg,
                                samples_per_epoch=max(2000, len(split.train) * 80))
            loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
            train_triplet(encoder, loader, device=device, epochs=args.epochs, lr=args.lr)
        else:
            from src.ptbdb_benchmark import PairDataset, train_siamese
            ds = PairDataset(repo, split.train, mode=args.seg,
                             samples_per_epoch=max(2000, len(split.train) * 80))
            loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
            train_siamese(encoder, loader, device=device, epochs=args.epochs, lr=args.lr)

        # Collect scores on test split (more samples = smoother curves)
        print("\nCollecting genuine/impostor scores...")
        profiles = make_profiles(encoder, repo, split.test,
                                 mode=args.seg, n_enroll=5, device=device)
        scores, labels = collect_scores(
            encoder, repo, profiles, split.test,
            mode=args.seg,
            genuine_per_user=20,
            impostor_per_user=20,
            device=device,
        )

        np.savez(args.save_scores, scores=scores, labels=labels)
        print(f"Scores saved → {args.save_scores}  "
              f"(genuine={labels.sum()}  impostor={(labels==0).sum()})")

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_fig8(scores, labels, out_path=args.out)


if __name__ == "__main__":
    main()
