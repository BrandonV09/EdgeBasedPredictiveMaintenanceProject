
#this is the feature extraction script for the CWRU data 

print("RUNNING make_features.py v2 from:", __file__)

# src/make_features.py  — minimal writer
import os
from glob import glob
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
import pandas as pd

FS = 12000
WIN = 2048
HOP = WIN // 2
NORMAL_GLOB = 'data/cwru/12k/normal/*.mat'
FAULT_GLOB  = 'data/cwru/12k/fault_drive_end/IR021_*.mat'   # change if IR014_* or B021_*
OUT_CSV     = 'results/features_12k_IR021.csv'
CHANNEL_SUFFIX = '_DE_time'

os.makedirs('results', exist_ok=True)

def load_signal(p):
    d = loadmat(p)
    key = next(k for k in d.keys() if k.endswith(CHANNEL_SUFFIX))
    return np.asarray(d[key]).squeeze()

def windows(x):
    for s in range(0, len(x) - WIN + 1, HOP):
        yield x[s:s+WIN]

def dominant_freq(w):
    w = w - np.mean(w)
    X = np.fft.rfft(w)
    mag = np.abs(X)
    idx = np.argmax(mag[1:]) + 1
    freqs = np.fft.rfftfreq(len(w), d=1.0/FS)
    return float(freqs[idx])

def features(w):
    w = w - np.mean(w)
    rms = float(np.sqrt(np.mean(w*w)))
    ptp = float(np.max(w) - np.min(w))
    krt = float(kurtosis(w, fisher=False))
    dom = dominant_freq(w)
    return rms, ptp, krt, dom

def process(path, label):
    x = load_signal(path)
    rows = []
    for w in windows(x):
        rms, ptp, krt, dom = features(w)
        rows.append({
            'file': os.path.basename(path),
            'label': int(label),
            'rms': rms,
            'ptp': ptp,
            'kurtosis': krt,
            'dominant_hz': dom
        })
    return rows

def main():
    normals = sorted(glob(NORMAL_GLOB))
    faults  = sorted(glob(FAULT_GLOB))
    if not normals: print('No normal files found:', NORMAL_GLOB)
    if not faults:  print('No fault files found:', FAULT_GLOB)

    all_rows = []
    for f in normals: all_rows += process(f, 0)
    for f in faults:  all_rows += process(f, 1)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'✅ Wrote {len(df)} rows to {OUT_CSV}')
    print(df.head())

if __name__ == "__main__":
    main()
