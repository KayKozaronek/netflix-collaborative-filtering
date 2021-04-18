"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    mu, var, p = mixture
    K = mu.shape[0]
    
    # Mask
    mask = X.astype(bool).astype(int)
    
    exp_term = (np.sum(X**2, axis=1)[:,None] + (mask @ mu.T**2) - 2*(X @ mu.T))/(2*var)
    factor_term = (-np.sum(mask, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    
    normal = factor_term - exp_term
    
    f_u_j = normal + np.log(p + 1e-16)
    
    logsums = logsumexp(f_u_j, axis=1).reshape(-1,1)
    log_posts = f_u_j- logsums
    
    LL = np.sum(logsums, axis=0)
    
    return np.exp(log_posts), LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu, _, _ = mixture
    K = mu.shape[0]
    
    # Update probabilities
    p = np.sum(post, axis=0)/n
    
    # Create mask
    mask = X.astype(bool).astype(int)
    
    # Update means 
    denom = post.T @ mask # Denominator (K,d)
    numer = post.T @ X  # Numerator (K,d)
    update_indices = np.where(denom >= 1)   # Indices for update
    mu[update_indices] = numer[update_indices] / denom[update_indices] 
    
    # Update variances
    denom_var = np.sum(post*np.sum(mask, axis=1).reshape(-1,1), axis=0) # (K,)
    
    norm = np.sum(X**2, axis=1)[:, np.newaxis] + (mask @ mu.T**2) - 2*(X @ mu.T)
        
    var = np.maximum(np.sum(post*norm, axis=0) / denom_var, min_variance)  
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_LL, LL = None, None
    
    while old_LL == None or LL - old_LL > 10**(-6) * np.abs(LL):
        old_LL = LL
        
        post, LL = estep(X, mixture) 
        mixture = mstep(X, post, mixture)

    return (mixture, post, LL)


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X[:]
    mu, var, p = mixture

    post, _ = estep(X, mixture)

    missing = np.where(X==0)

    X_pred[missing] = (post @ mu)[missing]

    return X_pred
