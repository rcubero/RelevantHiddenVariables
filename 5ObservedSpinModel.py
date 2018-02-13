# Some basic imports
from __future__ import division

import numpy as np
import pickle as pkl

from spin_operators import *
from boltzmann_learning import *
from optimal_partitioning import *

# full model implementation
configurations = [ np.array([ 2*int(binary(mu, N, i))-1 for i in np.arange(N) ]) for mu in range(2**N) ]
clusters = { binary(mu, N) : set([ i for i in range(N) if binary(mu, N, i) != '0']) for mu in range(2**N) }
averages = { mu : float(sum([phi(mu,s) for s in data])) / M for mu in clusters }
full_avg = {mu : averages[mu] for mu in averages if len(clusters[mu]) == 1 or len(clusters[mu]) == 2 }
full_coup=infer(full_avg,options={'gtol': 1e-6, 'disp': True})

f = open("/Users/rcubero/Dropbox/FindingRelevantHiddenVariable/5ObservedSpinModel/full_model_couplings.pkl","wb")
pkl.dump(full_coup,f); f.close()

# generate samples from the full model
partition_function = sum([ np.exp( np.sum([phi(mu, s, clusters)*full_coup[mu] for mu in full_coup]) ) for s in configurations ])
probabilities = [ np.exp( np.sum([phi(mu, s, clusters)*full_coup[mu] for mu in full_coup]) )/partition_function for s in configurations ]
test_data_index = np.random.choice(np.arange(len(configurations)),p=probabilities, size=1000)
test_data = np.array([configurations[index] for index in test_data_index])
#np.savetxt("/Users/rcubero/Dropbox/FindingRelevantHiddenVariable/5ObservedSpinModel/test_data.d", test_data)

# partially observed model implementation
test_partial_data = test_data[:, 0:5]
test_M, test_N = test_partial_data.shape

observed_configs = [ np.array([ 2*int(binary(mu, test_N, i))-1 for i in np.arange(test_N) ]) for mu in np.arange(2**test_N) ]
obs_clusters = { binary(mu, test_N) : set([ i for i in np.arange(test_N) if binary(mu, test_N, i) != '0']) for mu in range(2**test_N) }
test_partial_averages = { mu : float(sum([phi(mu, s, obs_clusters) for s in test_partial_data])) / test_M for mu in obs_clusters }
test_partial_avg = {mu : test_partial_averages[mu] for mu in test_partial_averages if len(obs_clusters[mu]) >= 1 }
test_partial_coup=infer(test_partial_avg, obs_clusters, observed_configs, options={'gtol': 1e-5, 'disp': True})

f = open("/Users/rcubero/Dropbox/FindingRelevantHiddenVariable/5ObservedSpinModel/test_partial_couplings.pkl","wb")
pkl.dump(test_partial_coup,f); f.close()

mq, kq, sq = optimal_partition(test_partial_data, a=1.0)
optimal_partition_coup = { mu:  optimal_partition_couplings(mu, test_N, sq, mq, kq, test_M, obs_clusters) for mu in test_partial_coup}
f = open("/Users/rcubero/Dropbox/FindingRelevantHiddenVariable/5ObservedSpinModel/optimal_partition_couplings.pkl","wb")
pkl.dump(optimal_partition_coup,f); f.close()
