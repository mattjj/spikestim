from __future__ import division
from scipy.io import loadmat


getfield = lambda name: lambda lst: map(lambda obj: getattr(obj, name), lst)
neurons = getfield('neuron')
allspikes = getfield('spikeCountsAllTrials')


def load():
    '''
    Data has a field called 'stimulus' that contains 15 fields, one for each distinct stimulus.
    For each stimulus there are 167 fields, one for each neuron.
    For each neuron there are 3 variables:
        - spikeCountsAll trials, a vector of 10 spike counts, one response for each trial;
        - spikeCountsMean;
        - spikeCountsVar.
    '''
    return loadmat('data/data.mat', squeeze_me=True, struct_as_record=False)['data']


def load_all_spikecounts():
    '''
    Returns a list of 15 stimuli, each element is a list of 167 neurons, each a
    list of 10 trails.

        data = load_all_spikecounts()
        for stim_idx, neurons in enumerate(data):
            for neuron_idx, trials in enumerate(neurons):
                print 'Avg. response from neuron {} to stimulus {}: {}'\
                    .format(neuron_idx, stim_idx, np.mean(trials))
    '''
    raw = load()
    return map(allspikes, neurons(raw.stimulus))
