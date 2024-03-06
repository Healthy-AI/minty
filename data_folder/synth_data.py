from util.utils import *
import numpy as pd
import pandas as pd
from util.missingness import Missingness


# Data generating process
'''
def generate_data(n=None, d=None, p=None, q=None):
    
    Generate data from a linear model with missing values
    n: number of samples
    d: number of features
    p: probability of missingness
    q: probability of change

    # Generate toy data - old version
    Xo = 1 * (np.random.randn(n, d) > 0)

    change = np.random.rand(n, 1) < q
    Xo[:, 4:5] = Xo[:, 0:1] * (1 - change) + (1 - Xo[:, 0:1]) * change
    Y = Xo[:, 0:1] * 2 + Xo[:, 4:5] * 2 + 1 + np.random.randn(n, 1)

    #Choose different missingness mechanisms
    X_m, M = Missingness().produce_NA(Xo, p_miss=0.1, mecha="MNAR", opt='logistic', p_obs=0.2, q=None)
    #X_m, M = Missingness().produce_NA(Xo, p_miss=0.1, mecha="MAR", opt=None, p_obs=0.2, q=0.2)

    #M = np.random.rand(n, d) < p
    #X_m = Xo.copy().astype(float)
    #X_m[M == 1] = np.nan
    return X_m, Y, M'''


def generate_data(n=None, c=None, delta=None, gamma=None):

    #n = 200
    #c = 3
    d = 2 * c
    s = 1
    #delta = 0.1  # Replacement disagreement probability
    #gamma = 0.4  # Missingness probability

    # Outcome coefficients
    beta = np.random.randn(d, 1)

    # First half of coefficients
    Xi = 1 * (np.random.randn(n, c) < 0)

    # Generate replacements
    f = 1 * (np.random.rand(n, c) < delta)
    Xj = Xi * (1 - f) + (1 - Xi) * f

    # Construct full vecctor
    X = np.hstack([Xi, Xj])

    # Construct outcome
    Y = np.dot(X, beta) + np.random.randn(n, 1) * s

    # Choose different missingness mechanisms
    X_m, M = Missingness().produce_NA(X, p_miss=0.1, mecha="MNAR", opt='logistic', p_obs=0.2, q=None)
    #X_m, M = Missingness().produce_NA(X, p_miss=0.1, mecha="MAR", opt=None, p_obs=0.2, q=0.2)

    #MCAR missingness
    # Missingness mask (only one of replacements ever observed)
    '''M = (np.random.rand(n, d) < gamma).astype(int)
    M2 = M.copy()
    M2[:, :c] = M[:, :c] - M[:, c:]
    M2[:, c:] = M[:, c:] - M[:, :c]
    M2[M2 < 0] = 0
    M2[M2 > 1] = 1
    M = M2'''

    X_m = X.copy().astype(float)
    X_m[M == 1] = np.nan

    return X_m, Y, M