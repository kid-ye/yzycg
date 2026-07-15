import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.preprocessing as pp

# Select two patients
patient1_path = "data/ptbdb/patient001/s0010_re"
patient2_path = "data/ptbdb/patient002/s0015lre"

print("Loading patient data...")

# Load raw signals
try:
    sig1 = pp.load_record(patient1_path)[0]  # First channel
    sig2 = pp.load_record(patient2_path)[0]
    print(f"Patient 1 signal length: {len(sig1)}")
    print(f"Patient 2 signal length: {len(sig2)}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Take first 2000 samples (10 seconds at 200Hz)
sig1_segment = sig1[:2000]
sig2_segment = sig2[:2000]

# Also get preprocessed versions
print("\nPreprocessing signals...")
preprocessed1 = pp.preprocess(sig1, fs=200, method='npd')
preprocessed2 = pp.preprocess(sig2, fs=200, method='npd')

if preprocessed1 and preprocessed2:
    prep1 = preprocessed1[0]
    prep2 = preprocessed2[0]
else:
    print("Preprocessing failed")
    sys.exit(1)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('ECG Waveform Comparison: Two Different Patients', fontsize=16, fontweight='bold')

# Time axes
time_raw = np.arange(len(sig1_segment)) / 200  # 200 Hz
time_prep = np.arange(1000) / 200  # 1000 samples

# Patient 1 - Raw
axes[0, 0].plot(time_raw, sig1_segment, 'b-', linewidth=0.8)
axes[0, 0].set_title('Patient 1 - Raw ECG', fontweight='bold')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# Patient 1 - Preprocessed
axes[0, 1].plot(time_prep, prep1, 'b-', linewidth=0.8)
axes[0, 1].set_title('Patient 1 - Preprocessed (Filtered & Normalized)', fontweight='bold')
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Normalized Amplitude')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Patient 2 - Raw
axes[1, 0].plot(time_raw, sig2_segment, 'r-', linewidth=0.8)
axes[1, 0].set_title('Patient 2 - Raw ECG', fontweight='bold')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)

# Patient 2 - Preprocessed
axes[1, 1].plot(time_prep, prep2, 'r-', linewidth=0.8)
axes[1, 1].set_title('Patient 2 - Preprocessed (Filtered & Normalized)', fontweight='bold')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Normalized Amplitude')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('patient_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'patient_comparison.png'")

# Also create an overlay comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle('ECG Overlay Comparison', fontsize=16, fontweight='bold')

# Raw overlay
axes2[0].plot(time_raw, sig1_segment, 'b-', linewidth=1, label='Patient 1', alpha=0.7)
axes2[0].plot(time_raw, sig2_segment, 'r-', linewidth=1, label='Patient 2', alpha=0.7)
axes2[0].set_title('Raw ECG Signals', fontweight='bold')
axes2[0].set_xlabel('Time (seconds)')
axes2[0].set_ylabel('Amplitude')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Preprocessed overlay
axes2[1].plot(time_prep, prep1, 'b-', linewidth=1, label='Patient 1', alpha=0.7)
axes2[1].plot(time_prep, prep2, 'r-', linewidth=1, label='Patient 2', alpha=0.7)
axes2[1].set_title('Preprocessed ECG Signals', fontweight='bold')
axes2[1].set_xlabel('Time (seconds)')
axes2[1].set_ylabel('Normalized Amplitude')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
axes2[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('patient_overlay.png', dpi=300, bbox_inches='tight')
print("✓ Overlay plot saved as 'patient_overlay.png'")

# Print statistics
print("\n=== Signal Statistics ===")
print(f"\nPatient 1 (Raw):")
print(f"  Mean: {sig1_segment.mean():.2f}")
print(f"  Std:  {sig1_segment.std():.2f}")
print(f"  Range: [{sig1_segment.min():.2f}, {sig1_segment.max():.2f}]")

print(f"\nPatient 2 (Raw):")
print(f"  Mean: {sig2_segment.mean():.2f}")
print(f"  Std:  {sig2_segment.std():.2f}")
print(f"  Range: [{sig2_segment.min():.2f}, {sig2_segment.max():.2f}]")

print(f"\nPatient 1 (Preprocessed):")
print(f"  Mean: {prep1.mean():.2f}")
print(f"  Std:  {prep1.std():.2f}")
print(f"  Range: [{prep1.min():.2f}, {prep1.max():.2f}]")

print(f"\nPatient 2 (Preprocessed):")
print(f"  Mean: {prep2.mean():.2f}")
print(f"  Std:  {prep2.std():.2f}")
print(f"  Range: [{prep2.min():.2f}, {prep2.max():.2f}]")

plt.show()
