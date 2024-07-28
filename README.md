# EEG Feature Extraction

This repository contains implementations of feature extraction methods for EEG data, specifically Common Spatial Patterns (CSP), Time Domain Parameters (TDP), and Power Spectral Density (PSD). These methods are widely used in the field of Brain-Computer Interfaces (BCI) for motor imagery tasks, and they have been tested for binary and multiclass discrimination tasks.

## Overview

Electroencephalogram (EEG) signals are a predominant source of neurophysiological data used in motor imagery-based brain-computer interfaces (MI-BCIs). Accurate discrimination of these signals is crucial for controlling external devices, such as robotic arms, based on user intent. This project provides implementations of three conventional EEG feature extraction techniques:

1. **Common Spatial Patterns (CSP)**: Extracts spatial features that maximize variance for one class while minimizing it for another.
2. **Time Domain Parameters (TDP)**: Uses Hjorth parameters to describe the signal in terms of activity, mobility, and complexity.
3. **Power Spectral Density (PSD)**: Estimates the power spectral density of the signal using Welch's method.

These methods have been evaluated for their performance in binary, ternary, and quaternary classification tasks involving complex motor imagery activities.

## Methods

### Common Spatial Patterns (CSP)
The CSP method is used to extract spatial features that maximize the variance for one class while minimizing it for another.

### Time Domain Parameters (TDP)
The TDP method uses Hjorth parameters to describe the signal in terms of activity, mobility, and complexity.

### Power Spectral Density (PSD)
The PSD method uses Welch's method to estimate the power spectral density of the signal.

## Usage

Example usage of each method is provided in the `examples` directory.

```python
# example_usage.py
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
