import wfdb
import matplotlib.pyplot as plt
import os

# Load a sample record from the dataset
record_path = os.path.join("data", "ptbdb", "patient001", "s0010_re")
record = wfdb.rdrecord(record_path, channels=[0, 1, 2])

# Extract signals
signals = record.p_signal
fs = record.fs  # sampling frequency
channels = record.sig_name

# Plot the first 3 channels for the first 3 seconds
duration = 3  # seconds
num_samples = int(fs * duration)

time_axis = [i / fs for i in range(num_samples)]

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
fig.suptitle(f"Sample ECG Data: patient001 (s0010_re)", fontsize=14, fontweight="bold")

for i in range(min(3, signals.shape[1])):
    axes[i].plot(time_axis, signals[:num_samples, i], color="tab:red")
    axes[i].set_ylabel(f"{channels[i]}")
    axes[i].grid(True)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()

out_img = os.path.join("results", "sample_data_plot.png")
plt.savefig(out_img)
plt.close(fig)
print(f"Sample data successfully plotted to {out_img}")
