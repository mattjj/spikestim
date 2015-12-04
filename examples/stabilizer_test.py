from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from spikestim import make_anscombe_stabilizer, make_laubscher_stabilizer, \
    negbin_var


if __name__ == "__main__":
    plt.figure(figsize=(11,8))

    for r in [2, 5, 10, 20]:
        # write p as a function of mu
        p = lambda mu: mu / (mu + r)

        # make stabilizers
        anscombe = make_anscombe_stabilizer(r)
        laubscher = make_laubscher_stabilizer(r)

        # for a range of means, compute corresponding variance
        means = np.linspace(0.01, 50., 100)
        anscombe_vars = [negbin_var(anscombe, r, p(mu)) for mu in means]
        laubscher_vars = [negbin_var(laubscher, r, p(mu)) for mu in means]

        # plot the result
        line, = plt.plot(means, anscombe_vars, label='anscombe, r={}'.format(r))
        plt.plot(means, laubscher_vars, line.get_color() + '--', label='laubscher, r={}'.format(r))

    # label the plot
    plt.xlabel(r'$\mathbb{E}[X]$')
    plt.ylabel(r'Var$[f(X)]$')
    plt.title(r'Variance of $f(X)$ as a function of $\mu = \mathbb{E}[X]$,  '
              r'$X \sim \mathrm{NB}\left(r, \; p=\frac{\mu}{\mu + r}\right)$')
    plt.legend(loc='best')

    plt.show()
