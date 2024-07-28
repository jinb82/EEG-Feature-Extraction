import numpy as np
from scipy.linalg import eigh

def compute_covariance_matrix(data):
    return np.dot(data, data.T) / data.shape[1]

def csp(X, y, num_components=2):
    """
    Compute the Common Spatial Patterns (CSP)
    
    Parameters:
    X -- EEG data, shape (n_samples, n_channels, n_times)
    y -- labels, shape (n_samples,)
    num_components -- number of spatial filters to return for each class
    
    Returns:
    W -- CSP filters, shape (n_channels, 2 * num_components)
    """
    class_labels = np.unique(y)
    X1 = X[y == class_labels[0]]
    X2 = X[y == class_labels[1]]
    
    C1 = np.mean([compute_covariance_matrix(epoch) for epoch in X1], axis=0)
    C2 = np.mean([compute_covariance_matrix(epoch) for epoch in X2], axis=0)
    
    C = C1 + C2
    E, U = eigh(C1, C)
    
    indices = np.empty(len(E), dtype=int)
    indices[0::2] = np.arange(len(E) // 2)
    indices[1::2] = np.arange(len(E) - 1, len(E) // 2 - 1, -1)
    
    W = U[:, indices]
    
    return W[:, :num_components].T, W[:, -num_components:].T
