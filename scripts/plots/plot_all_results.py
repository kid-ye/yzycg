import numpy as np
import matplotlib.pyplot as plt
import os

THRESHOLDS = np.linspace(-1.0, 1.0, 1000)

def far_frr(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    far = fp / (fp + tn + 1e-10)
    frr = fn / (fn + tp + 1e-10)
    return far, frr

def compute_curves(scores, labels):
    fars, frrs = [], []
    for t in THRESHOLDS:
        f, r = far_frr(scores, labels, t)
        fars.append(f)
        frrs.append(r)
    return np.array(fars), np.array(frrs)

def eer(fars, frrs):
    diff = np.abs(fars - frrs)
    idx  = np.argmin(diff)
    return THRESHOLDS[idx], (fars[idx] + frrs[idx]) / 2

files = [
    ("scores_npd_siamese.npz", "NPD - Siamese"),
    ("scores_npd_triplet.npz", "NPD - Triplet"),
    ("scores_p2t_siamese.npz", "P2T - Siamese"),
    ("scores_p2t_triplet.npz", "P2T - Triplet"),
    ("scores_r2r_siamese.npz", "R2R - Siamese"),
    ("scores_r2r_triplet.npz", "R2R - Triplet")
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("FAR and FRR for All Architectures and Losses", fontsize=16, fontweight="bold")

for ax, (fname, title) in zip(axes.flat, files):
    path = os.path.join("results", fname)
    if os.path.exists(path):
        data = np.load(path)
        # Handle cases where keys might be slightly different
        if 'scores' in data and 'labels' in data:
            scores, labels = data['scores'], data['labels']
        else:
            # Maybe the default keys like arr_0, arr_1 
            keys = list(data.keys())
            scores, labels = data[keys[0]], data[keys[1]]
            
        fars, frrs = compute_curves(scores, labels)
        eer_t, eer_val = eer(fars, frrs)

        ax.plot(THRESHOLDS, fars * 100, color="tab:red", linewidth=1.8, label="FAR")
        ax.plot(THRESHOLDS, frrs * 100, color="tab:blue", linewidth=1.8, label="FRR")
        
        ax.axvline(x=eer_t, color="gray", linestyle=":", linewidth=1.0)
        ax.plot(eer_t, eer_val * 100, "ko", markersize=6, label=f"EER={eer_val*100:.2f}%")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Rate (%)", fontsize=10)
        ax.set_xlim(-0.1, 1.05)
        ax.set_ylim(-1, 101)
        ax.legend(fontsize=9)
        print(f"Processed {fname}: EER = {eer_val*100:.2f}%")
    else:
        ax.set_title(f"{title} (File not found)", fontsize=12)
        print(f"Not found: {fname}")

plt.tight_layout()
out_img = os.path.join("results", "all_models_far_frr.png")
plt.savefig(out_img)
print(f"Saved plot to {out_img}")
