# src/main.py
from scipy.io import loadmat
from glob import glob
import os
import matplotlib.pyplot as plt

# Helpful: see where Python thinks the "current folder" is
print("Working directory:", os.getcwd())

# Paths to your .mat files (relative to project root)
NORMAL_GLOB = 'data/cwru/12k/normal/Normal_*.mat'
FAULT_GLOB  = 'data/cwru/12k/fault_drive_end/IR021_*.mat'  # or IR014_*.mat, B021_*.mat

def load_drive_end_signal(mat_path):
    d = loadmat(mat_path)
    # pick the first key that ends with '_DE_time' (drive-end channel)
    key = next(k for k in d.keys() if k.endswith('_DE_time'))
    return d[key].squeeze()

# Find files
normal_files = sorted(glob(NORMAL_GLOB))
fault_files  = sorted(glob(FAULT_GLOB))

print(f"Found {len(normal_files)} normal files and {len(fault_files)} fault files")

# Load first normal and first fault to sanity-check
if normal_files:
    sig_n = load_drive_end_signal(normal_files[0])
    print("Normal example length:", len(sig_n))
    plt.figure(); plt.plot(sig_n); plt.title(f"Normal: {os.path.basename(normal_files[0])}"); plt.show()

if fault_files:
    sig_f = load_drive_end_signal(fault_files[0])
    print("Fault example length:", len(sig_f))
    plt.figure(); plt.plot(sig_f); plt.title(f"Fault: {os.path.basename(fault_files[0])}"); plt.show()
