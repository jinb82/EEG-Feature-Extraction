import numpy as np
from scipy.signal import welch

def psd(X, fs=1000, nperseg=256):
    """
    Compute Power Spectral Density (PSD) for EEG data
    
    Parameters:
    X -- EEG data, shape (n_samples, n_channels, n_times)
    fs -- sampling frequency
    nperseg -- length of each segment for Welch's method
    
    Returns:
    features -- PSD features, shape (n_samples, n_channels * (nperseg // 2 + 1))
    """
    n_samples, n_channels, n_times = X.shape
    features = []
    
    for i in range(n_samples):
        sample_features = []
        for j in range(n_channels):
            f, Pxx = welch(X[i, j], fs=fs, nperseg=nperseg)
            sample_features.extend(Pxx)
        features.append(sample_features)
    
    return np.array(features)
