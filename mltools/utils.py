################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np

from numpy import asarray as arr
from numpy import atleast_2d as twod



def checkDataShape(X,Y):
    """
    Simple helper function to convert vectors to matrices and check the shape of
    the data matrices X,Y
    """
    X = twod(X).T if X.ndim < 2 else X
    #Y = twod(Y).T if Y.ndim < 2 else Y
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y do not have the same number of data points!")
    return X,Y

################################################################################
## Class to index and index to one-hot (one-of-K) transorms ####################
################################################################################


def to1ofK(Y, values=None):
    """
    Function that converts Y into discrete valued matrix;
    i.e.: to1ofK([3,3,2,2,4,4]) = [[ 1 0 0 ]
                                   [ 1 0 0 ]
                                   [ 0 1 0 ]
                                   [ 0 1 0 ]
                                   [ 0 0 1 ]
                                   [ 0 0 1 ]]

    Parameters
    ----------
    Y : array like
        1 x N (or N x 1) array of values (ints) to be converted.
    values : list (optional)
        List that specifices indices of of Y values in return matrix.

    Returns
    -------
    array
        Discrete valued 2d representation of Y.
    """
    n,d = np.matrix(Y).shape

    assert min(n,d) == 1
    values = list(values) if values is not None else list(np.unique(Y))
    C = len(values)
    flat_Y = Y.flatten()
   
    index = []
    for l in flat_Y:
        index.append(values.index(l))

    return np.array([[0 if r != i else 1 for i in range(C)] for r in index])



def from1ofK(Y, values=None):
    """
    Function that converts Y from 1-of-K ("1-hot") rep back to single col/row form.

    Parameters
    ----------
    Y : arraylike
        Matrix to convert from 1-of-k rep.
    values : list (optional)
        List that specifies which values to use for which index.

    Returns
    -------
    array
        Y in single row/col form.
    """
    #return Y.argmax(1) if values is None else twod([values[i] for i in Y.argmax(1)]).T
    return Y.argmax(1) if values is None else arr(values)[Y.argmax(1)]; 


def toIndex(Y, values=None):
    """
    Function that converts discrete value Y into [0 .. K - 1] (index) 
    representation; i.e.: toIndex([4 4 1 1 2 2], [1 2 4]) = [2 2 0 0 1 1].

    Parameters
    ----------
    Y      : (M,) or (M,1) array-like of values to be converted
    values : optional list that specifices the value/index mapping to use for conversion.

    Returns
    -------
    idx    : (M,) or (M,1) array that contains indexes instead of values.
    """
    n,d = np.matrix(Y).shape

    assert min(n,d) == 1
    values = list(values) if values is not None else list(np.unique(Y))
    C = len(values)
    #flat_Y = Y.ravel()

    idx = []
    for v in Y:
        idx.append(values.index(v))
    return np.array(idx)



def fromIndex(Y, values):
    """
    Convert index-valued Y into discrete representation specified by values
    in values.

    Parameters
    ----------
    Y : numpy array
        1 x N (or N x 1) numpy array of indices.
    values : numpy array
        1 x max(Y) array of values for conversion.

    Returns
    -------
    discrete_Y : numpy array
        1 x N (or N x 1) numpy array of discrete values.
    """
    discrete_Y = arr(values)[arr(Y)]
    return discrete_Y



################################################################################
## Basic data set operations: shuffle, split, xval, bootstrap ##################
################################################################################

def shuffleData(X, Y=None):
    """
    Shuffle (randomly reorder) data in X and Y.

    Parameters
    ----------
    X : MxN numpy array: N feature values for each of M data points
    Y : Mx1 numpy array (optional): target values associated with each data point

    Returns
    -------
    X,Y  :  (tuple of) numpy arrays of shuffled features and targets
            only returns X (not a tuple) if Y is not present or None
    
    Ex:
    X2    = shuffleData(X)   : shuffles the rows of the data matrix X
    X2,Y2 = shuffleData(X,Y) : shuffles rows of X,Y, preserving correspondence
    """
    nx,dx = twod(X).shape
    Y = arr(Y).flatten()
    ny = len(Y)

    pi = np.random.permutation(nx)
    X = X[pi,:]

    if ny > 0:
        assert ny == nx, 'shuffleData: X and Y must have the same length'
        Y = Y[pi] if Y.ndim <= 1 else Y[pi,:]
        return X,Y

    return X


def splitData(X, Y=None, train_fraction=0.80):
    """
    Split data into training and test data.

    Parameters
    ----------
    X : MxN numpy array of data to split
    Y : Mx1 numpy array of associated target values
    train_fraction : float, fraction of data used for training (default 80%)

    Returns
    -------
    to_return : (Xtr,Xte,Ytr,Yte) or (Xtr,Xte)
        A tuple containing the following arrays (in order): training
        data from X, testing data from X, training labels from Y
        (if Y contains data), and testing labels from Y (if Y 
        contains data).
    """
    nx,dx = twod(X).shape
    ne = int(round(train_fraction * nx))

    Xtr,Xte = X[:ne,:], X[ne:,:]
    to_return = (Xtr,Xte)

    if Y is not None:
        Y = arr(Y).flatten()
        ny = len(Y)
        if ny > 0:
            assert ny == nx, 'splitData: X and Y must have the same length'
            Ytr,Yte = Y[:ne], Y[ne:]
            to_return += (Ytr,Yte)

    return to_return


def crossValidate(X, Y=None, K=5, i=0):
    """
    Create a K-fold cross-validation split of a data set:
    crossValidate(X,Y, 5, i) : return the ith of 5 80/20 train/test splits

    Parameters
    ----------
    X : MxN numpy array of data points to be resampled.
    Y : Mx1 numpy array of labels associated with each datum (optional)
    K : number of folds of cross-validation
    i : current fold to return (0...K-1)

    Returns
    -------
    Xtr,Xva,Ytr,Yva : (tuple of) numpy arrays for the split data set
    If Y is not present or None, returns only Xtr,Xva
    """
    nx,dx = twod(X).shape
    start = int(round( float(nx)*i/K ))
    end   = int(round( float(nx)*(i+1)/K ))

    Xte   = X[start:end,:] 
    Xtr   = np.vstack( (X[0:start,:],X[end:,:]) )
    to_return = (Xtr,Xte)

    Y = arr(Y).flatten()
    ny = len(Y)

    if ny > 0:
        assert ny == nx, 'crossValidate: X and Y must have the same length'
        if Y.ndim <= 1:
            Yte = Y[start:end]
            Ytr = np.hstack( (Y[0:start],Y[end:]) )
        else:   # in case targets are multivariate
            Yte = Y[start:end,:]
            Ytr = np.vstack( (Y[0:start,:],Y[end:,:]) )
        to_return += (Ytr,Yte)

    return to_return



def bootstrapData(X, Y=None, n_boot=None):
    """
    Bootstrap resample a data set (with replacement): 
    draw data points (x_i,y_i) from (X,Y) n_boot times.

    Parameters
    ----------
    X : MxN numpy array of data points to be resampled.
    Y : Mx1 numpy array of labels associated with each datum (optional)
    n_boot : int, number of samples to draw (default: M)

    Returns
    -------
    Xboot, Yboot : (tuple of) numpy arrays for the resampled data set
    If Y is not present or None, returns only Xboot (non-tuple)
    """
    nx,dx = twod(X).shape
    if n_boot is None: n_boot = nx
    idx = np.floor(np.random.rand(n_boot) * nx).astype(int)
    if Y is None: return X[idx,:]
    Y = Y.flatten()
    assert nx == len(Y), 'bootstrapData: X and Y should have the same length'
    return (X[idx,:],Y[idx])



################################################################################
################################################################################
################################################################################
