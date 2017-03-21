import numpy as np

from .base import classifier
from .base import regressor
from numpy import asarray as arr
from numpy import asmatrix as mat


################################################################################
## KNNCLASSIFY #################################################################
################################################################################


class knnClassify(classifier):
    """A k-nearest neighbor classifier

    Attributes:
        Xtr,Ytr : training data (features and target classes)
        classes : a list of the possible class labels
        K       :  the number of neighbors to use in the prediction
                alpha   : the (inverse) "bandwidth" for a weighted prediction
                     0 = use unweighted data in the prediction
                     a = weight data point xi proportional to exp( - a * |x-xi|^2 ) 
    """

    def __init__(self, X=None, Y=None, K=1, alpha=0):
        """Constructor for knnClassify object.  

        Any parameters are passed directly to train(); see train() for arguments.
        """
        self.K = K
        self.Xtr = []
        self.Ytr = []
        self.classes = []
        self.alpha = alpha

        if type(X) == np.ndarray and type(Y) == np.ndarray:
            self.train(X, Y)


    def __repr__(self):
        str_rep = 'knn classifier, {} classes, K={}{}'.format(
            len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


    def __str__(self):
        str_rep = 'knn classifier, {} classes, K={}{}'.format(
            len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


## CORE METHODS ################################################################
            

    def train(self, X, Y, K=None, alpha=None):
        """Train the classifier (for knn: store the input data)

        Args:
          X (arr): MxN array of M training examples with N features each
          Y (arr): M, or M,1 array of target values associated with each datum
          K (int): The number of neighbors to use for predictions.
          alpha (float): Nonzero => use weighted average, Gaussian kernel with inverse scale alpha
        """
        self.Xtr = np.asarray(X)
        self.Ytr = np.asarray(Y)
        self.classes = list(np.unique(Y))
        if K is not None:
            self.K = K
        if alpha is not None:
            self.alpha = alpha


    def predictSoft(self, X):
        """This method makes a "soft" nearest-neighbor prediction on test data.

        Args:
            X (array): M,N array of M data points of N features to predict with

        Returns:
            P (array): M,C array of C class probabilities for each data point
        """
        mtr,ntr = arr(self.Xtr).shape      # get size of training data
        mte,nte = arr(X).shape                 # get size of test data
        if nte != ntr:
            raise ValueError('Training and prediction data must have same number of features')
        
        num_classes = len(self.classes)
        prob = np.zeros((mte,num_classes))     # allocate memory for class probabilities
        K = min(self.K, mtr)                   # (can't use more neighbors than training data points)
        for i in range(mte):                   # for each test example...
            # ...compute sum of squared differences...
            dist = np.sum(np.power(self.Xtr - arr(X)[i,:], 2), axis=1)
            # ...find nearest neighbors over training data and keep nearest K data points
            indices = np.argsort(dist, axis=0)[0:K]             
            sorted_dist = dist[indices];       # = np.sort(dist, axis=0)[0:K]                
            wts = np.exp(-self.alpha * sorted_dist)
            count = np.zeros((num_classes,));
            for c in range(len(self.classes)): # find total weight of instances of that classes
                count[c] = 1.0 * np.sum(wts[self.Ytr[indices] == self.classes[c]]);
            prob[i,:] = count / count.sum()    # save (soft) results
        return prob

    #def predict(self, X):
    #    """Not implemented; uses predictSoft.  Implementation might be more efficient for large C"""
    #    mtr,ntr = arr(self.Xtr).shape      # get size of training data
    #    mte,nte = arr(X).shape                 # get size of test data
    #    assert nte == ntr, 'Training and prediction data must have same number of features'
    #    
    #    num_classes = len(self.classes)
    #    Y_te = np.tile(self.Ytr[0], (mte, 1))      # make Y_te same data type as Ytr
    #    K = min(self.K, mtr)                           # (can't use more neighbors than training data points)
    #    for i in range(mte):                           # for each test example...
    #        # ...compute sum of squared differences...
    #        dist = np.sum(np.power(self.Xtr - arr(X)[i,:], 2), axis=1)
    #        # ...find neares neighbors over training data and keep nearest K data points
    #        sorted_dist = np.sort(dist, axis=0)[0:K]
    #        indices = np.argsort(dist, axis=0)[0:K]
    #        wts = np.exp(-self.alpha * sorted_dist)
    #        count = []
    #        for c in range(len(self.classes)):
    #            # total weight of instances of that classes
    #            count.append(np.sum(wts[self.Ytr[indices] == self.classes[c]]))
    #        count = np.asarray(count)
    #        c_max = np.argmax(count)                   # find largest count...
    #        Y_te[i] = self.classes[c_max]              # ...and save results
    #    return Y_te


################################################################################
################################################################################
################################################################################


class knnRegress(regressor):
    """A k-nearest neighbor regressor

    Attributes:
        Xtr,Ytr : training data (features and target values)
        K       : the number of neighbors to use in the prediction
        alpha   : the (inverse) "bandwidth" for a weighted prediction
                     0 = use unweighted data in the prediction
                     a = weight data point xi proportional to exp( - a * |x-xi|^2 ) 
    """

    def __init__(self, X=None, Y=None, K=1, alpha=0):
        """Constructor for knnRegress object.  

        Any parameters are passed directly to train(); see train() for arguments.
        """
        self.K = K
        self.Xtr = []
        self.Ytr = []
        self.alpha = alpha

        if X is not None and Y is not None:
            self.train(X, Y)


    def __repr__(self):
        str_rep = 'knnRegress, K={}{}'.format(
            self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


    def __str__(self):
        str_rep = 'knnRegress, K={}{}'.format(
            self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


## CORE METHODS ################################################################
            

    def train(self, X, Y, K=None, alpha=None):
        """Train the regressor (for knn: store the input data)

        Args:
          X (arr): MxN array of M training examples with N features each
          Y (arr): M, or M,1 array of target values associated with each datum
          K (int): The number of neighbors to use for predictions.
          alpha (float): Nonzero => use weighted average, Gaussian kernel with inverse scale alpha
        """
        self.Xtr = np.asarray(X)
        self.Ytr = np.asarray(Y)
        if K is not None:
            self.K = K
        if alpha is not None:
            self.alpha = alpha



    def predict(self, X):
        """This method makes a nearest neighbor prediction on test data X.
    
        Args:
          X : MxN numpy array containing M data points with N features each

        Returns:
          array : M, or M,1 numpy array of the predictions for each data point
        """
        ntr,mtr = arr(self.Xtr).shape              # get size of training data
        nte,mte = arr(X).shape                         # get size of test data

        if mtr != mte:
            raise ValueError('knnRegress.predict: training and prediction data must have the same number of features')

        Y_te = np.tile(self.Ytr[0], (nte, 1))     # make Y_te the same data type as Ytr
        K = min(self.K, ntr)                          # can't have more than n_tr neighbors

        for i in range(nte):
            dist = np.sum(np.power((self.Xtr - X[i]), 2), axis=1)  # compute sum of squared differences
            sorted_idx = np.argsort(dist, axis=0)[:K]         # find nearest neihbors over Xtr and...
            sorted_dist = dist[sorted_idx];                   # ...keep nearest K data points
            wts = np.exp(-self.alpha * sorted_dist)
            Y_te[i] = arr(wts).dot(self.Ytr[sorted_idx].T) / np.sum(wts)  # weighted average

        return Y_te





