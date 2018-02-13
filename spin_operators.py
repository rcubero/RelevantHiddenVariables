from __future__ import division

import numpy as np
import operator, functools

# binary representation from integer
def binary(mu, N, i=None):
    """ binary representation from integer """
    to_bin = ("{0:0"+str(N)+"b}").format(mu)
    if i==None:
        return to_bin
    else:
        return to_bin[i]

# defines the operators
def phi(mu, s, clusters):
    return functools.reduce(operator.mul, [s[i] for i in clusters[mu]], 1)
