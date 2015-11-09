from __future__ import division, print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import grad
import scipy.optimize


'''
The code in this file implements a method for finding a stationary point of
the negative binomial likelihood via Newton's method, described here:
https://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation

The main function to use is fit_maxlike(data, r_guess)
'''


def newton(f, x0):
    # wrap scipy.optimize.newton with our automatic derivatives
    return scipy.optimize.newton(f, x0, fprime=grad(f), fprime2=grad(grad(f)))


def negbin_loglike(r, p, x):
    # the negative binomial log likelihood we want to maximize
    return gammaln(r+x) - gammaln(r) - gammaln(x+1) + x*np.log(p) + r*np.log(1-p)


def negbin_sample(r, p, size):
    # a negative binomial is a gamma-compound-Poisson
    return npr.poisson(npr.gamma(r, p/(1-p), size=size))


def fit_maxlike(data, r_guess):
    # follows Wikipedia's section on negative binomial max likelihood
    assert np.var(data) > np.mean(data), "Likelihood-maximizing parameters don't exist!"
    loglike = lambda r, p: np.sum(negbin_loglike(r, p, data))
    p = lambda r: np.sum(data) / np.sum(r+data)
    rprime = lambda r: grad(loglike)(r, p(r))
    r = newton(rprime, r_guess)
    return r, p(r)
