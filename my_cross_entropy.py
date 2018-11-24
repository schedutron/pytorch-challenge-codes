from math import log
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -sum([Y[i]*log(P[i]) + (1-Y[i])*log(1-P[i]) for i in range(len(P))])