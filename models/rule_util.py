import numpy as np
import scipy
import time

def rule_assignments(X, S):
    """ 
    Computes the assignments of observations in X to rules in S
    
    args: 

    X : numpy array of shape (m x d) where m is the number of observations and d the number of features
    S : numpy array of shape (d x k) where k is the number of rules and d the number of features

    Note: 
    - Expects X to have an intercept column at [:,0] 
    """

    a = 1*(np.dot(X, S)>0)

    return a  
    
def missingness_reliance(X, M, S):
    """
    Returns the reliance in S on columns of X with missing values in M 

    The returned value is the average number of rules that have no observed value across both rules and observations
    
    args: 
    X : features of shape (n x d)
    M : missingness mask of shape (n x d)
    S : rule matrix of shape (d x k)
    
    """

    n = X.shape[0]
    k = S.shape[1]

    if k == 0:
        return np.zeros((n, 1))
    
    rho = np.zeros((n, k))
    for r in range(k):
        Sr = S[:, r]
        rho[:,r] = (1-np.max((Sr>0) & (1-M) & (X==1), axis=1)) * np.max((Sr>0) & M, axis=1)            

    return rho