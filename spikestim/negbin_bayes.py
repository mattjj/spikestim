from __future__ import division
import numpy as np

from pybasicbayes.distributions import NegativeBinomial


'''
The code in this file provides utilities for Bayesian estimation of negative
binomial parameters through MCMC methods provided by pybasicbayes.

The main function to use is get_posterior_samples(data, num_samples).

The NB class sets the prior to use sensible defaults, namely

    p ~ Beta(alpha=1., beta=1.)
    r ~ Gamma(k=1., theta=1)

That is, the prior on p is uniform on [0,1] and the prior on r is exponential
with rate 1.
'''


class NB(NegativeBinomial):
    def __init__(self, r=None, p=None, alpha=1., beta=1., k=1., theta=1.):
        super(NB, self).__init__(
            r=r, p=p, alpha_0=alpha, beta_0=beta, k_0=k, theta_0=theta)


def get_posterior_samples(data, num_samples):
    distn = NB()
    data = np.require(data, requirements='C')
    samples = []
    for _ in xrange(num_samples):
        distn.resample(data)
        samples.append((distn.r, distn.p))
    return samples


# these next two functions are redundant with negbin_maxlike.py, but use
# pybasicbayes implementations instead


def negbin_loglike(r, p, x):
    return NB(r=r, p=p).log_likelihood(x)


def negbin_sample(r, p, size):
    return NB(r=r, p=p).rvs(size)
