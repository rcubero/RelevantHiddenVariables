from __future__ import division

import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool

from spin_operators import *

# defines the likelihoods.
# the model is specified implicitly by the keys of `couplings` and `averages`
def calc_partition_term(zipped_data):
    phi, couplings, s, clusters = zipped_data
    return np.exp( np.sum([phi(mu, s, clusters)*couplings[mu] for mu in couplings]) )

def calc_energy_term(zipped_data):
    couplings, averages, mu = zipped_data
    return couplings[mu]*averages[mu]

def log_lik_slow(couplings, averages, clusters, configurations):
    assert '0' * N not in couplings
    assert set(couplings.keys()) == set(averages.keys())
    
    # partition function
    pool = Pool()
    res = pool.map_async(calc_partition_term,[(phi, couplings, s, clusters) for s in configurations])
    pool.close(); pool.join()
    Z = sum(res.get())
    
    # energy
    pool = Pool()
    res = pool.map_async(calc_energy_term,[(couplings, averages, mu) for mu in couplings])
    pool.close(); pool.join()
    E = - sum(res.get())
    
    return - np.log(Z) - E

# defines the observables conjugated with the couplings.
# the model is specified implicitly by the keys of `couplings`
def obs(couplings, clusters, configurations):
    assert '0' * N not in couplings
    # weights
    weights = np.array([ np.exp( np.sum([phi(mu, s, clusters)*couplings[mu] for mu in couplings]) ) for s in configurations ])
    # partition function
    Z = np.sum(weights)
    # observables
    observables = { mu : sum([
                              phi(mu, configurations[i], clusters) * weights[i] for i in np.arange(len(configurations))
                              ]) / Z for mu in couplings }
    return observables

# inference of best couplings
def infer(averages, clusters, configurations, options={'gtol': 1e-3, 'disp': True}):
    assert '0' * N not in averages
    
    # the model to use
    operators = list(averages.keys())
    n=len(operators)
    # function to maximize
    g0 = np.zeros(n)
    def target(g):
        return -log_lik_slow({ operators[mu] : g[mu] for mu in range(n) }, averages, clusters, configurations)
    # calculates result
    res = minimize(target, g0, method='BFGS',options=options)
    g = res.x
    
    return { operators[mu] : g[mu] for mu in range(n)}
