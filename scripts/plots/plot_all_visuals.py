import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb
from sklearn.metrics import roc_curve, auc

os.makedirs("results", exist_ok=True)

files = [
    ("scores_npd_siamese.npz", "NPD - Siamese"),
    ("scores_npd_triplet.npz", "NPD - Triplet"),
    ("scores_p2t_siamese.npz", "P2T - Siamese"),
    ("scores_p2t_triplet.npz", "P2T - Triplet"),
    ("scores_r2r_siamese.npz", "R2R - Siamese"),
    ("scores_r2r_triplet.npz", "R2R - Triplet")
]

# --- 1. Genuine vs Impostor Score Histograms ---
print("Generating Plot 1: Score Histograms...")
fig_hist, axes_hist = plt.subplots(2, 3, figsize=(16, 9))
fig_hist.suptitle("Score Distributions: Genuine vs Impostor", fontsize=16, fontweight="bold")

for ax, (fname, title) in zip(axes_hist.flat, files):
    path = os.path.join("results", fname)
    if os.path.exists(path):
        data = np.load(path)
        scores, labels = (data['scores'], data['labels']) if 'scores' in data else (data[list(data.keys())[0]], data[list(data.keys())[1]])
        
        gen_scores = scores[labels == 1]
        imp_scores = scores[labels == 0]
        
        ax.hist(gen_scores, bins=50, alpha=0.6, color='tab:blue', density=True, label='Genuine (Same Person)')
        ax.hist(imp_scores, bins=50, alpha=0.6, color='tab:red', density=True, label='Impostor (Different Person)')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Similarity Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/01_score_histograms.png")
plt.close(fig_hist)


# --- 2. ROC Curves ---
print("Generating Plot 2: ROC Curves...")
plt.figure(figsize=(9, 7))

for fname, title in files:
    path = os.path.join("results", fname)
    if os.path.exists(path):
        data = np.load(path)
        scores, labels = (data['scores'], data['labels']) if 'scores' in data else (data[list(data.keys())[0]], data[list(data.keys())[1]])
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{title} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/02_roc_curves.png")
plt.close()


# --- 3. Intra-Patient vs. Inter-Patient ECG ---
print("Generating Plot 3: Intra vs Inter Patient...")
p1_dir = os.path.join("data", "ptbdb", "patient001")
p2_dir = os.path.join("data", "ptbdb", "patient002")

def get_records(p_dir):
    if not os.path.exists(p_dir): return []
    return sorted([f.replace('.hea', '') for f in os.listdir(p_dir) if f.endswith('.hea')])

p1_recs = get_records(p1_dir)
p2_recs = get_records(p2_dir)

if len(p1_recs) >= 2 and len(p2_recs) >= 1:
    rec_paths = [
        (os.path.join(p1_dir, p1_recs[0]), "Patient 1 - Record A"),
        (os.path.join(p1_dir, p1_recs[1]), "Patient 1 - Record B (Different day/time)"),
        (os.path.join(p2_dir, p2_recs[0]), "Patient 2 - Record A")
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Intra-Patient vs. Inter-Patient ECG Comparison (Lead I)", fontsize=14, fontweight="bold")
    
    for i, (rpath, title) in enumerate(rec_paths):
        # Read only Lead I
        record = wfdb.rdrecord(rpath, channels=[0])
        fs = record.fs
        duration = 2.5 # Plot 2.5 seconds
        sig = record.p_signal[:int(fs * duration), 0]
        time = np.arange(len(sig)) / fs
        
        color = 'tab:blue' if i < 2 else 'tab:orange'
        axes[i].plot(time, sig, color=color, lw=1.5)
        axes[i].set_title(title, fontsize=12)
        axes[i].set_ylabel("mV", fontsize=10)
        axes[i].grid(True)
        
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/03_intra_vs_inter.png")
    plt.close(fig)


# --- 4. Full 12-Lead ECG ---
print("Generating Plot 4: 12-Lead ECG...")
if len(p1_recs) >= 1:
    rec_path_12 = os.path.join(p1_dir, p1_recs[0])
    try:
        # 12 channels correspond to indices 0 through 11
        record12 = wfdb.rdrecord(rec_path_12, channels=list(range(12)))
        duration = 2.5
        sig12 = record12.p_signal[:int(record12.fs * duration), :]
        time12 = np.arange(sig12.shape[0]) / record12.fs
        
        fig, axes = plt.subplots(6, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(f"Full 12-Lead ECG: patient001 (Record {p1_recs[0]})", fontsize=16, fontweight="bold")
        
        channels = record12.sig_name
        
        for i in range(12):
            ax = axes[i % 6, i // 6]
            ax.plot(time12, sig12[:, i], color='tab:green', lw=1.2)
            ax.set_ylabel(channels[i], fontweight="bold", fontsize=10)
            ax.grid(True)
            if i % 6 == 5:
                ax.set_xlabel("Time (s)", fontsize=10)
                
        plt.tight_layout()
        plt.savefig("results/04_full_12_lead.png")
        plt.close(fig)
    except Exception as e:
        print(f"Error on 12-Lead: {e}")

print("All plots generated successfully!")