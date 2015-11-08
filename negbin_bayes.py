from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import NegativeBinomial


'''
The code in this file provides a class for Bayesian estimation of negative
binomial parameters through MCMC methods provided by pybasicbayes. The NB class
sets the prior to use sensible defaults, namely

    p ~ Beta(alpha=1., beta=1.)
    r ~ Gamma(k=1., theta=1)

That is, the prior on p is uniform on [0,1] and the prior on r is exponential
with rate 1.
'''


class NB(NegativeBinomial):
    def __init__(self, r=None, p=None, alpha=1., beta=1., k=1., theta=1.):
        super(NB, self).__init__(
            r=r, p=p, alpha_0=alpha, beta_0=beta, k_0=k, theta_0=theta)


def get_samples(data, num_samples):
    distn = NB()
    data = np.require(data, requirements='C')
    samples = []
    for _ in xrange(num_samples):
        distn.resample(data)
        samples.append((distn.r, distn.p))
    return samples


def negbin_loglike(r, p, x):
    return NB(r=r, p=p).log_likelihood(x)


def plot_samples(data, samples):
    xm = data.max()
    plt.figure()
    plt.hist(data, bins=np.arange(xm+1)-0.5, normed=True)
    plt.xlim(0, xm)

    for r, p in samples:
        plt.plot(np.arange(xm), np.exp(negbin_loglike(r, p, np.arange(xm))))

    plt.xlabel('k')
    plt.ylabel('p(k)')


if __name__ == "__main__":
    # generate data
    npr.seed(0)
    data = NB(r=5, p=0.5).rvs(1000)

    # run MCMC to generate parameter samples
    samples = get_samples(data, 100)

    # plot results
    every_10_samples = samples[9::10]
    plot_samples(data, every_10_samples)

    plt.show()
