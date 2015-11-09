from __future__ import division

from spikestim.negbin_maxlike import negbin_sample, fit_maxlike
from spikestim.stabilizing_transforms import make_anscombe_stabilizer

if __name__ == "__main__":
    # generate data
    data = negbin_sample(r=5, p=0.5, size=1000)

    # fit a model to the data
    r, p = fit_maxlike(data, 3.)

    # make a stabilizing function using the fit model
    stabilize = make_anscombe_stabilizer(r)

    # apply the stabilizer to the data
    trasnformed_data = stabilize(data)
