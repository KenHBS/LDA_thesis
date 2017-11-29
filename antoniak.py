# Sample from Antoniak Distribution with Python. `rand_antoniak`
# draws a sample from the distribution of tables created by a
# Chinese restaurant process with parameter `alpha` after `n`
# patrons are seated. Some notes on this distribution are
# here: http://www.cs.cmu.edu/~tss/antoniak.pdf.

import numpy as np
from numpy.random import choice
from scipy.special import gamma
from random import choices

def stirling(N, m):
    if N < 0 or m < 0:
        raise Exception("Bad input to stirling.")
    if m == 0 and N > 0:
        return 0
    elif (N, m) == (0, 0):
        return 1
    elif N == 0 and m > 0:
        return m
    elif m > N:
        return 0
    else:
        return stirling(N-1, m-1) + (N-1) * stirling(N-1, m)

def normalized_stirling_numbers(nn):
    #  * stirling(nn) Gives unsigned Stirling numbers of the first
    #  * kind s(nn,*) in ss. ss[i] = s(nn,i). ss is normalized so that maximum
    #  * value is 1. After Teh (npbayes).
    ss = [stirling(nn, i) for i in range(1, nn + 1)]
    max_val = max(ss)
    return np.array(ss, dtype=float) / max_val

def rand_antoniak(param, mm, stirling_matrix):
    mm = int(mm)
    #p = normalized_stirling_numbers(int(mm))
    p = stirling_matrix[mm, :(mm+1)]
    for i, m in enumerate(p):
        g_num = gamma(param)
        g_den = gamma(param + mm)
        p[i] = (g_num / g_den) * p[i] * param ** m
    p /= np.sum(p)
    a = np.array(range(0, int(mm) + 1))
    return choices(a, p)[0]

def rand_antoniak_2(param, mm):
    p = normalized_stirling_numbers(int(mm))
    for i, m in enumerate(p):
        g_num = gamma(param)
        g_den = gamma(param + mm)
        p[i] = (g_num/g_den) * p[i] * param**m
    p /= p.sum()
    a = np.array(range(1, int(mm)+1))
    return choice(a=a, p=p)

def rand_antoniak_short(param, mm):
    p = normalized_stirling_numbers(int(mm))
    p /= p.sum()
    a = np.array(range(1, int(mm)+1))
    return(choice(a=a, p=p))

