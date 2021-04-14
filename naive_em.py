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
    log_likelihood = 

    return soft_counts, log_likelihood

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
    n, K = post.shape
    n, d = X.shape
    
    post_sum = np.sum(post, axis=0)
    
    mu_hat = np.dot(post.T, X) / post_sum.reshape(K,1)
    p_hat = 1/n * post_sum
    
    norm = [np.sum(post[j] * np.linalg.norm(X - mu_hat[j])**2) for j in range(K)]
    var_hat = norm / (d* post_sum) 
    
    gaussian_mixture = GaussianMixture(mu_hat, var_hat, p_hat)

    return gaussian_mixture


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
    raise NotImplementedError

    # convergence
    while not new_ll - old_ll <= 10**(-6) * np.abs(new_ll):


    return (gaussian_mixture, post, LL)
