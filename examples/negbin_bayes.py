from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from spikestim import negbin_loglike, negbin_sample, get_posterior_samples


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
    data = negbin_sample(r=5, p=0.5, size=1000)

    # run MCMC to generate parameter samples
    samples = get_posterior_samples(data, 100)

    # plot results
    every_10_samples = samples[9::10]
    plot_samples(data, every_10_samples)

    plt.show()
