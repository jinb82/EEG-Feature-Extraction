import numpy as np
from csp import csp
from tdp import tdp
from psd import psd

# Example EEG data
X = np.random.randn(100, 64, 256)  # 100 samples, 64 channels, 256 time points
y = np.random.randint(0, 2, 100)   # Binary labels

# CSP
W1, W2 = csp(X, y)
print("CSP filters computed.")

# TDP
tdp_features = tdp(X)
print("TDP features computed.")

# PSD
psd_features = psd(X)
print("PSD features computed.")
