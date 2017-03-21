## IMPORTS #####################################################################
import math
import numpy as np

from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from numpy import asmatrix as arr

from .utils import toIndex


################################################################################
## Base (abstract) "classify" class and associated functions ###################
################################################################################

class classifier:

  def __init__(self, *args, **kwargs):
    """Constructor for abstract base class for various classifiers. 

    This class implements methods that generalize to different classifiers.
    Optional arguments X,Y,... call train(X,Y,...) to initialize the model
    """
    self.classes = []
    # TODO: if Y!=None init classes from data? (leave to train?)
    if len(args) or len(kwargs):
        return self.train(*args, **kwargs)


  def __call__(self, *args, **kwargs):
    """Provides syntatic sugar for prediction; calls "predict".  """ 
    return self.predict(*args, **kwargs)


  def predict(self, X):
    """Abstract method, implemented by derived classes.

    Args:
        X (arr): M,N array of M data points with N features each

    Returns:
        arr: M, or M,1 array of the predicted class for each data point

    Derived classes do not need to implement this function if predictSoft is
    implemented; by default it uses predictSoft and converts to the most likely class.
    """
    idx = np.argmax( self.predictSoft(X) , axis=1 )      # find most likely class (index)
    return np.asarray(self.classes)[idx]                 # convert to saved class values


  def predictSoft(self,X):
    """Abstract method, implemented by derived classes.

    Args:
        X (arr): M,N array of M data points with N features each

    Returns:
        arr: MxC array of C class probabilities for each data point
    """
    raise NotImplementedError

  ####################################################
  # Standard loss f'n definitions for classifiers    #
  ####################################################
  def err(self, X, Y):
    """This method computes the error rate on a data set (X,Y)

    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        float: fraction of prediction errors, 1/M \sum (Y[i]!=f(X[i]))
    """
    Y    = arr( Y )
    Yhat = arr( self.predict(X) )
    return np.mean(Yhat.reshape(Y.shape) != Y)


  def nll(self, X, Y):
    """Compute the (average) negative log-likelihood of the soft predictions 

    Using predictSoft, normalizes and inteprets as conditional probabilities to compute
      (1/M) \sum_i log Pr[ y^{(i)} | f, x^{(i)} ]

    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        float: Negative log likelihood of the predictions
    """
    M,N = X.shape
    P = arr( self.predictSoft(X) )
    P /= np.sum(P, axis=1, keepdims=True)       # normalize to sum to one
    Y = toIndex(Y, self.classes)
    return - np.mean( np.log( P[ np.arange(M), Y ] ) ) # evaluate



  def auc(self, X, Y):
    """Compute the area under the roc curve on the given test data.

    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        float: Area under the ROC curve

    This method only works on binary classifiers. 
    """
    if len(self.classes) != 2:
      raise ValueError('This method can only supports binary classification ')

    try:                  # compute 'response' (soft binary classification score)
      soft = self.predictSoft(X)[:,1]  # p(class = 2nd)
    except (AttributeError, IndexError):  # or we can use 'hard' binary prediction if soft is unavailable
      soft = self.predict(X)

    n,d = twod(soft).shape             # ensure soft is the correct shape
    soft = soft.flatten() if n==1 else soft.T.flatten()

    indices = np.argsort(soft)         # sort data by score value
    Y = Y[indices]
    sorted_soft = soft[indices]

    # compute rank (averaged for ties) of sorted data
    dif = np.hstack( ([True],np.diff(sorted_soft)!=0,[True]) )
    r1  = np.argwhere(dif).flatten()
    r2  = r1[0:-1] + 0.5*(r1[1:]-r1[0:-1]) + 0.5
    rnk = r2[np.cumsum(dif[:-1])-1]

    # number of true negatives and positives
    n0,n1 = sum(Y == self.classes[0]), sum(Y == self.classes[1])

    if n0 == 0 or n1 == 0:
      raise ValueError('Data of both class values not found')

    # compute AUC using Mann-Whitney U statistic
    result = (np.sum(rnk[Y == self.classes[1]]) - n1 * (n1 + 1.0) / 2.0) / n1 / n0
    return result


  def confusion(self, X, Y):
    """Estimate the confusion matrix (Y x Y_hat) from test data.
    
    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        C (arr): C[i,j] = # of data from class i that were predicted as class j
    """
    Y_hat = self.predict(X)
    num_classes = len(self.classes)
    indices = toIndex(Y, self.classes) + num_classes * (toIndex(Y_hat, self.classes) - 1)
    C = np.histogram(indices, np.arange(1, num_classes**2 + 2))[0]
    C = np.reshape(C, (num_classes, num_classes))
    return np.transpose(C)


  def roc(self, X, Y):
    """Compute the receiver operating charateristic curve on a data set.

    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        tuple : (fpr,tpr,tnr) where 
                fpr = false positive rate (1xN numpy vector)
                tpr = true positive rate (1xN numpy vector)
                tnr = true negative rate (1xN numpy vector)

    This method is only defined for binary classifiers. 
    Plot fpr vs. tpr to see the ROC curve. 
    Plot tpr vs. tnr to see the sensitivity/specificity curve.
    """
    if len(self.classes) > 2:
      raise ValueError('This method can only supports binary classification ')

    try:                  # compute 'response' (soft binary classification score)
      soft = self.predictSoft(X)[:,1]  # p(class = 2nd)
    except (AttributeError, IndexError):
      soft = self.predict(X)        # or we can use 'hard' binary prediction if soft is unavailable
    n,d = twod(soft).shape

    if n == 1:
      soft = soft.flatten()
    else:
      soft = soft.T.flatten()

    # number of true negatives and positives
    n0 = float(np.sum(Y == self.classes[0]))
    n1 = float(np.sum(Y == self.classes[1]))

    if n0 == 0 or n1 == 0:
      raise ValueError('Data of both class values not found')

    # sort data by score value
    indices = np.argsort(soft)
    Y = Y[indices]
    sorted_soft = soft[indices] #np.sort(soft)

    # compute false positives and true positive rates
    tpr = np.divide(np.cumsum(Y[::-1] == self.classes[1]).astype(float), n1)
    fpr = np.divide(np.cumsum(Y[::-1] == self.classes[0]).astype(float), n0)
    tnr = np.divide(np.cumsum(Y == self.classes[0]).astype(float), n0)[::-1]

    # find ties in the sorting score
    same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)
    tpr = np.append([0], tpr[np.logical_not(same)])
    fpr = np.append([0], fpr[np.logical_not(same)])
    tnr = np.append([1], tnr[np.logical_not(same)])
    return fpr, tpr, tnr



################################################################################
## REGRESS #####################################################################
################################################################################


class regressor:

  def __init__(self, *args, **kwargs):
    """Simple constructor for base regressor class; specialized by various learners"""
    if len(args) or len(kwargs):
        return self.train(*args, **kwargs)



  def __call__(self, *args, **kwargs):
    """Syntatic sugar for prediction; same as "predict".  """ 
    return self.predict(*args, **kwargs)



  ####################################################
  # Standard loss f'n definitions for regressors     #
  ####################################################
  def mae(self, X, Y):
    """Computes the mean absolute error

    Computes
      (1/M) \sum_i | f(x^{(i)}) - y^{(i)} |
    of a regression model f(.) on test data X and Y. 

    Args:
      X (arr): M x N array that contains M data points with N features
      Y (arr): M x 1 array of target values for each data point

    Returns:
      float: mean absolute error
    """
    Yhat = self.predict(X)
    return np.mean(np.absolute(Y - Yhat.reshape(Y.shape)), axis=0)


  def mse(self, X, Y):
    """Computes the mean squared error

    Computes
      (1/M) \sum_i ( f(x^{(i)}) - y^{(i)} )^2 
    of a regression model f(.) on test data X and Y. 

    Args:
      X (arr): M x N array that contains M data points with N features
      Y (arr): M x 1 array of target values for each data point

    Returns:
      float: mean squared error
    """
    Yhat = self.predict(X)
    return np.mean( (Y - Yhat.reshape(Y.shape))**2 , axis=0)


  def rmse(self, X, Y):
    """Computes the root mean squared error
  
    Computes
      sqrt( f.mse(X,Y) )
    of a regression model f(.) on test data X and Y. 

    Args:
      X (arr): M x N array that contains M data points with N features
      Y (arr): M x 1 array of target values for each data point

    Returns:
      float: root mean squared error
    """
    return np.sqrt(self.mse(X, Y))



################################################################################
################################################################################
################################################################################
