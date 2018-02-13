from __future__ import division

import numpy as np
from mpmath import log, loggamma, psi, mp
from collections import Counter

from spin_operators import *

DPS = 20
mp.dps = DPS
mp.pretty = True

def likelihood(Kq1,Kq2,mq1,mq2,Nq,M,a):
    Kq1 = float(Kq1); Kq2 = float(Kq2); mq1 = float(mq1); mq2 = float(mq2); Nq = float(Nq); M = float(M); a = float(a);
    term1 = loggamma(Kq1+Kq2+a) + loggamma(a) - loggamma(Kq1+a) - loggamma(Kq2+a)
    term2 = float(Kq1)*log(float(mq1)/float(mq1+mq2)) + float(Kq2)*log(float(mq2)/float(mq1+mq2))
    term3 = loggamma(M+(a*Nq)) + loggamma(a*(Nq-1)) - loggamma(M+(a*(Nq-1))) - loggamma(a*Nq)
    return term1+term2+term3

def histo(histogram):
    return np.delete(np.bincount(histogram),0)

def mqkq(mk):
    kq = np.array([(i+1)*mk[i] for i in range(len(mk))])
    return mk[mk>0], kq[kq>0]

def merging_process(mq,kq,M,a,states):
    possibility = True
    while possibility is True:
        if len(kq)>1:
            delta_L = [likelihood(kq[i],kq[i+1],mq[i],mq[i+1],len(kq),M,a) for i in range(len(kq)-1)]
            if np.amax(delta_L)>0:
                index = np.where(delta_L == np.amax(delta_L))[0]
                if len(index)>1: max_ind = np.random.choice(index,1)
                else: max_ind = np.argmax(delta_L)
                mq[max_ind] = mq[max_ind]+mq[max_ind+1]
                kq[max_ind] = kq[max_ind]+kq[max_ind+1]
                states[max_ind] = np.append(states[max_ind], states[max_ind+1])
                mq = np.delete(mq,max_ind+1)
                kq = np.delete(kq,max_ind+1)
                states = np.delete(states,max_ind+1)
            else:
                possibility = False
        else:
            possibility = False
    return mq, kq, states

def s2i_states(data):
    # transform data from {-1,+1} to {0, 1}
    binary_data = (data+1)/2
    
    # binary state to integer state
    binary_state = binary_data.dot(1 << np.arange(binary_data.shape[-1] - 1, -1, -1))
    binary_state = binary_state.astype('int')
    
    return binary_state

def optimal_partition(data, a=1.0):
    # transforms data (spin configuration states) into binary integers
    binary_state = s2i_states(data)
    
    # make S-partition
    data_histogram = Counter(binary_state)
    data_histogram = np.array(data_histogram.most_common())
    
    # make K-partition
    mk = histo(data_histogram[:,1])
    # get all of the states seen the same number of times
    sq = [data_histogram[:,0][np.where(data_histogram[:,1]==i)[0]] for i in np.where(mk>0)[0]+1]
    # configure the K-partitions in a way compatible with merging_process
    mq, kq = mqkq(mk)
    
    # returns Q*-partition
    return merging_process(mq,kq,M,a,sq)

def optimal_partition_couplings(mu, N, sq, mq, kq, M, clusters):
    return np.sum([np.sum([phi(mu, np.array([ 2*int(binary(s, N, i))-1 for i in np.arange(N) ]), clusters) for s in sq[j]])*np.log(kq[j]/(M*mq[j])) for j in np.arange(len(mq))])/np.power(2,N)
