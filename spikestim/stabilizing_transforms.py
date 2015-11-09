from __future__ import division
import numpy as np

from negbin_maxlike import negbin_loglike


def make_anscombe_stabilizer(r):
    def stabilize(x):
        return np.sqrt(r - 1./2) * np.arcsinh(np.sqrt((x + 3./8)/(r - 3./4)))
    return stabilize


def make_laubscher_stabilizer(r):
    def stabilize(x):
        return np.sqrt(r) * np.arcsinh(np.sqrt(x/r)) \
            + np.sqrt(r - 1)*np.arcsinh(np.sqrt((x + 3./4)/(r - 3./2)))
    return stabilize


def negbin_expectation(f, r, p, x_trunc=5000):
    "Compute E[f(X)] with X ~ NB(r,p) by truncating the PMF"
    pmf = lambda x: np.exp(negbin_loglike(r, p, x))
    x = np.arange(x_trunc)
    return np.dot(f(x), pmf(x))


def negbin_var(f, r, p, x_trunc=5000):
    "Compute Var[f(X)] with X ~ NB(r, p) by truncating the PMF"
    fsq = lambda x: f(x)**2
    E = lambda f: negbin_expectation(f, r, p, x_trunc)
    return E(fsq) - E(f)**2
