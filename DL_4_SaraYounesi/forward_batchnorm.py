from numpy import *
import numpy as np


def forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode):
    """
    Performs the forward propagation through a without_hNorm layer.

    Arguments:
    Z -- input, with shape (num_examples, num_features)
    gamma -- vector, BN layer parameter
    beta -- vector, BN layer parameter
    eps -- scalar, BN layer hyperparameter
    beta_avg -- scalar, beta value to use for moving averages
    mode -- boolean, indicating whether used at 'train' or 'test' time

    Returns:
    out -- output, with shape (num_examples, num_features)
    """

    if mode == 'train':
        # TODO: Mean of Z across first dimension
        mu = np.mean(Z, axis=0, keepdims=True)

        # TODO: Variance of Z across first dimension
        var = np.var(Z, axis=0, keepdims=True)

        # Take moving average for cache_dict['mu']
        cache_dict['mu'] = beta_avg * cache_dict['mu'] + (1-beta_avg) * mu

        # Take moving average for cache_dict['var']
        cache_dict['var'] = beta_avg * cache_dict['var'] + (1-beta_avg) * var

        # X = (X - mu) / np.sqrt(var + eps)
        # out = gamma * X + beta

    elif mode == 'test':
        # TODO: Load moving average of mu
        mu = cache_dict['mu']

        # TODO: Load moving average of var
        var = cache_dict['var']

     # TODO: Apply z_norm transformation
    # Z_norm =
    Z_norm = (Z - mu) / np.sqrt(var + eps)

    # TODO: Apply gamma and beta transformation to get Z tiled
    out = gamma * Z_norm + beta

    return out


# Practical Test


X_size = 10
b_avg = 0.7
epsilon = 0.0001
bias = 200
X = np.random.randn(5, 4) + bias


gamma = np.random.rand(1, 4)
beta = np.random.rand(1, 4)

cache_dict = {
    'mu': np.zeros((1, 4)),
    'var': np.zeros((1, 4))
}

mode = 'train'
applied = forward_batchnorm(
    X, gamma, beta, epsilon, cache_dict, b_avg, 'train')
print('')
print('________________________________________________________________________________')
var = np.var(applied, axis=0)
mu = np.mean(applied, axis=0)
print(var, mu)
print('________________________________________________________________________________')
print('')

print(cache_dict)
print('________________________________________________________________________________')
print('matris without')
print('________________________________________________________________________________')
print(X)
print('________________________________________________________________________________')
print('matris applied')
print('________________________________________________________________________________')
print(applied)
