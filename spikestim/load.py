from __future__ import division
from scipy.io import loadmat


def load():
    '''
    Data has a field called ‘stimulus’  that contains 15 fields, one for each distinct stimulus.
    For each stimulus there are 167 fields, one for each neuron.
    For each neuron there are 3 variables:
        - spikeCountsAll trials, a vector of 10 spike counts, one response for each trial;
        - spikeCountsMean;
        - spikeCountsVar.
    '''
    return loadmat('data/data.mat', squeeze_me=True, struct_as_record=False)
