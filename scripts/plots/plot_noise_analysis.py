import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import scipy.signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.preprocessing as pp

# Select a patient
patient_path = "data/ptbdb/patient001/s0010_re"

print("Loading patient data...")

# Load raw signal
try:
    sig_raw = pp.load_record(patient_path)[0]  # First channel
    print(f"Signal length: {len(sig_raw)}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Take first 2000 samples (10 seconds at 200Hz)
sig_segment = sig_raw[:2000]

# Apply bandpass filter manually to see what's filtered
LOW_CUT = 0.5
HIGH_CUT = 40.0
FS = 200.0

nyquist = 0.5 * FS
low = LOW_CUT / nyquist
high = HIGH_CUT / nyquist
b, a = scipy.signal.butter(4, [low, high], btype='band')
sig_filtered = scipy.signal.lfilter(b, a, sig_segment)

# Calculate the noise (what was removed)
noise = sig_segment - sig_filtered

# Also show frequency spectrum
from scipy.fft import fft, fftfreq

# FFT of raw signal
fft_raw = fft(sig_segment)
fft_filtered = fft(sig_filtered)
fft_noise = fft(noise)
freqs = fftfreq(len(sig_segment), 1/FS)

# Only positive frequencies
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
fft_raw_pos = np.abs(fft_raw[pos_mask])
fft_filtered_pos = np.abs(fft_filtered[pos_mask])
fft_noise_pos = np.abs(fft_noise[pos_mask])

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('ECG Signal Filtering Analysis: Noise Rejection Visualization', 
             fontsize=16, fontweight='bold')

time = np.arange(len(sig_segment)) / FS

# 1. Raw Signal
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time, sig_segment, 'b-', linewidth=0.8)
ax1.set_title('Raw ECG Signal (Unfiltered)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'Mean: {sig_segment.mean():.2f}\nStd: {sig_segment.std():.2f}', 
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Filtered Signal
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time, sig_filtered, 'g-', linewidth=0.8)
ax2.set_title('Filtered ECG Signal (0.5-40 Hz Bandpass)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Amplitude')
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f'Mean: {sig_filtered.mean():.2f}\nStd: {sig_filtered.std():.2f}', 
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 3. Rejected Noise
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time, noise, 'r-', linewidth=0.8)
ax3.set_title('Rejected Noise (Raw - Filtered)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Amplitude')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.text(0.02, 0.98, f'Mean: {noise.mean():.2f}\nStd: {noise.std():.2f}', 
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# 4. Overlay Comparison
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time, sig_segment, 'b-', linewidth=1, label='Raw Signal', alpha=0.6)
ax4.plot(time, sig_filtered, 'g-', linewidth=1.5, label='Filtered Signal', alpha=0.8)
ax4.set_title('Raw vs Filtered Overlay', fontweight='bold', fontsize=12)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Amplitude')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# 5. Frequency Spectrum - Full comparison
ax5 = fig.add_subplot(gs[2, :])
ax5.semilogy(freqs_pos, fft_raw_pos, 'b-', linewidth=1, label='Raw Signal', alpha=0.7)
ax5.semilogy(freqs_pos, fft_filtered_pos, 'g-', linewidth=1.5, label='Filtered Signal', alpha=0.8)
ax5.semilogy(freqs_pos, fft_noise_pos, 'r-', linewidth=1, label='Rejected Noise', alpha=0.7)
ax5.axvline(x=LOW_CUT, color='orange', linestyle='--', linewidth=2, label=f'Low Cutoff ({LOW_CUT} Hz)')
ax5.axvline(x=HIGH_CUT, color='purple', linestyle='--', linewidth=2, label=f'High Cutoff ({HIGH_CUT} Hz)')
ax5.set_title('Frequency Spectrum Analysis', fontweight='bold', fontsize=12)
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Magnitude (log scale)')
ax5.set_xlim([0, 100])
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3, which='both')

# Add annotations
ax5.text(0.25, 0.95, '← Baseline Wander\n(< 0.5 Hz)', 
         transform=ax5.transAxes, fontsize=9, color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax5.text(0.75, 0.95, 'High Freq Noise →\n(> 40 Hz)', 
         transform=ax5.transAxes, fontsize=9, color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.savefig('noise_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Noise analysis plot saved as 'noise_analysis.png'")

# Print detailed statistics
print("\n=== Filtering Statistics ===")
print(f"\nRaw Signal:")
print(f"  Mean: {sig_segment.mean():.4f}")
print(f"  Std:  {sig_segment.std():.4f}")
print(f"  Range: [{sig_segment.min():.4f}, {sig_segment.max():.4f}]")
print(f"  Energy: {np.sum(sig_segment**2):.2f}")

print(f"\nFiltered Signal:")
print(f"  Mean: {sig_filtered.mean():.4f}")
print(f"  Std:  {sig_filtered.std():.4f}")
print(f"  Range: [{sig_filtered.min():.4f}, {sig_filtered.max():.4f}]")
print(f"  Energy: {np.sum(sig_filtered**2):.2f}")

print(f"\nRejected Noise:")
print(f"  Mean: {noise.mean():.4f}")
print(f"  Std:  {noise.std():.4f}")
print(f"  Range: [{noise.min():.4f}, {noise.max():.4f}]")
print(f"  Energy: {np.sum(noise**2):.2f}")

noise_percentage = (np.sum(noise**2) / np.sum(sig_segment**2)) * 100
print(f"\nNoise Energy: {noise_percentage:.2f}% of total signal")
print(f"Signal Retained: {100 - noise_percentage:.2f}%")

# Frequency band analysis
low_freq_noise = np.sum(fft_noise_pos[freqs_pos < LOW_CUT]**2)
high_freq_noise = np.sum(fft_noise_pos[freqs_pos > HIGH_CUT]**2)
total_noise = np.sum(fft_noise_pos**2)

print(f"\n=== Noise Frequency Distribution ===")
print(f"Low frequency noise (< {LOW_CUT} Hz): {(low_freq_noise/total_noise)*100:.1f}%")
print(f"High frequency noise (> {HIGH_CUT} Hz): {(high_freq_noise/total_noise)*100:.1f}%")

plt.show()
