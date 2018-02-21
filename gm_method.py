# Some basic imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, operator,functools
import pickle as pkl

from scipy.optimize import minimize
from multiprocessing import Pool
from mpmath import log, loggamma, psi, mp

from spin_operators import *
from optimal_partitioning import *

# imports actual data
N=9
data = np.empty((0,N))
with open('../data/v.d','r') as file:
    for line in file:
        s = np.array([[2*int(c)-1 for c in line.strip()]])
        data = np.concatenate((data,s),axis=0)
M,N=data.shape
data.shape

# defines the configurations
observed_configs = [ np.array([ 2*int(binary(mu, N, i))-1 for i in np.arange(N) ]) for mu in np.arange(2**N) ]
# defines the clusters (as dictionaries)
obs_clusters = { binary(mu, N) : set([ i for i in np.arange(N) if binary(mu, N, i) != '0']) for mu in range(2**N) }
# defines the 2**N-1 operators needed in this study
operators = [mu for mu in list(obs_clusters.keys()) if len(obs_clusters[mu])>=1]


# create the K-partitions with the unseen states
M, N = data.shape
binary_state = s2i_states(data)
data_histogram = Counter(binary_state)
data_histogram = np.array(data_histogram.most_common())

mk = np.bincount(data_histogram[:,1])
mk[0] = np.power(2,N) - np.sum(mk)
sq = [data_histogram[:,0][np.where(data_histogram[:,1]==i)[0]] if i > 0 else np.setdiff1d(np.arange(np.power(2,N)), data_histogram[:,0]) for i in np.where(mk>0)[0]]

mq, kq = mqkq(mk[1:])
mq = np.append(mk[0], mq)
kq = np.append([0], kq)


# compute for the phi^mu(s) matrix
phi_matrix = np.array([ [ 1-2*np.mod(np.sum([int(btest(mu,i) and not btest(s,i)) for i in np.arange(N)]),2) for mu in np.arange(1, np.power(2,N)) ]  for s in np.arange(np.power(2,N)) ] )
# compute the chi matrix
Chi_matrix = np.array([ [ np.sum([phi_matrix[s][mu] for s in sq[j]]) for mu in np.arange(np.power(2,N)-1) ] for j in np.arange(len(kq))] )


# perform singular value decomposition
#   s = sorted singular values, size is the same as the number of partitions
#   v = the rows correspond to the singular vectors
#   u = the weight matrix
u, s, v = np.linalg.svd(Chi_matrix, full_matrices=0)


# plot the contribution of each partition to g_lambda
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig,ax = plt.subplots(dpi=600)
fig.set_size_inches(10,6)

for j in np.arange(len(kq)):
    ax.plot(np.arange(len(kq)), np.power([-v[j]@np.array(Chi_matrix)[i]/np.power(2,N) for i in np.arange(len(kq))],2), "o-", label=r'$\lambda_{%s}=$%.3e'%(str(int(j)),s[j]/np.power(2,N)))

my_xticks = ['$k$=%s,$m_k$=%s'%(str(int(kq[i]/mq[i])), str(mq[i])) for i in np.arange(len(kq))]

ax.set_title("$K$-Partition with Unseen States")
ax.set_xticks(np.arange(len(kq)))
ax.set_xticklabels(my_xticks, rotation='vertical', fontsize=14)
ax.set_xlabel(r'$k_q$', fontsize=16)
ax.set_ylabel(r'$\vert a_{q,\lambda} \vert^2$', fontsize=16)
ax.legend(loc="best")
plt.savefig("figure.pdf")
