from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import multigrad
import matplotlib.pyplot as plt

from spikestim import negbin_sample, fit_maxlike, negbin_loglike


def plot_results(data, r, p):
    xm = data.max()
    plt.figure()
    plt.hist(data, bins=np.arange(xm+1)-0.5, normed=True, label='normed data counts')
    plt.xlim(0,xm)
    plt.plot(np.arange(xm), np.exp(negbin_loglike(r, p, np.arange(xm))), label='maxlike fit')
    plt.xlabel('k')
    plt.ylabel('p(k)')
    plt.legend(loc='best')


if __name__ == "__main__":
    # generate data
    npr.seed(0)
    data = negbin_sample(r=5, p=0.5, size=1000)

    # fit likelihood-extremizing parameters
    r, p = fit_maxlike(data, r_guess=1)

    # report fit
    print('Fit parameters:')
    print('r={r}, p={p}'.format(r=r, p=p))

    print('Check that we are at a local stationary point:')
    loglike = lambda r, p: np.sum(negbin_loglike(r, p, data))
    grad_both = multigrad(loglike, argnums=[0,1])
    print(grad_both(r, p))

    plot_results(data, r, p)
    plt.show()
