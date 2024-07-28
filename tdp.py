import numpy as np

def hjorth_parameters(data):
    """
    Compute Hjorth parameters (Activity, Mobility, Complexity) for an EEG signal
    
    Parameters:
    data -- EEG signal, shape (n_times,)
    
    Returns:
    activity, mobility, complexity
    """
    activity = np.var(data)
    mobility = np.sqrt(np.var(np.diff(data)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(data))) / np.var(np.diff(data)) / mobility)
    
    return activity, mobility, complexity

def tdp(X):
    """
    Compute Time Domain Parameters (TDP) for EEG data
    
    Parameters:
    X -- EEG data, shape (n_samples, n_channels, n_times)
    
    Returns:
    features -- TDP features, shape (n_samples, n_channels * 3)
    """
    n_samples, n_channels, n_times = X.shape
    features = np.zeros((n_samples, n_channels * 3))
    
    for i in range(n_samples):
        for j in range(n_channels):
            activity, mobility, complexity = hjorth_parameters(X[i, j])
            features[i, j*3] = activity
            features[i, j*3+1] = mobility
            features[i, j*3+2] = complexity
    
    return features
