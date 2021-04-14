"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    mu, var, p = mixture
    
    K = mu.shape[0]
#    var = var.reshape(K,1)
#    p = p[:, np.newaxis]
    n, d = X.shape
    X = X[:, None]
    
    norm = np.linalg.norm(X - mu, axis=2) ** 2
    exp_term = np.exp(- 1/(2*var) * norm)
    factor_term = 1 / (2 * np.pi * var)**(d/2)
    
    normal = factor_term * exp_term
    
    denominator = np.sum(p * normal,axis=1).reshape(-1,1)
    post = p * normal / denominator
    
    LL = np.sum(np.log(denominator), axis=0)
    
    return post, LL

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    K = post.shape[1]
    n, d = X.shape
    
    post_sum = np.sum(post, axis=0)
    
    mu_hat = np.dot(post.T, X) / post_sum.reshape(K,1)
    p_hat = post_sum / n
    
    norm = np.linalg.norm(X[:,None] - mu_hat, axis=2)**2 
    var_hat = np.sum(post* norm, axis=0) / (d* post_sum) 
    
    gaussian_mixture = GaussianMixture(mu_hat, var_hat, p_hat)

    return gaussian_mixture



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture
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
        mixture = mstep(X, post)

    return (mixture, post, LL)