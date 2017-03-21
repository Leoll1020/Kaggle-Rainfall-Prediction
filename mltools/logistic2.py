import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat


################################################################################
## LOGISTIC REGRESSION CLASSIFIER ##############################################
################################################################################


class logisticClassify2(classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
                  (1xN numpy array, where N=# features)
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        self.theta = np.array([])

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


    def __repr__(self):
        str_rep = 'logisticClassify2 model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


    def __str__(self):
        str_rep = 'logisticClassify2 model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


## CORE METHODS ################################################################

    def plotBoundary(self,X,Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        M,N = X.shape
        if (M != 2): raise ValueError("Data and model must be 2-dimensional")
        raise NotImplementedError
        ## TODO: plot data (X[:,0] vs X[:,1], colored by class Y[:]
        ## TODO: plot decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with 
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0 
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        raise NotImplementedError
        ## TODO: compute linear response z[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] for each i
        ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
        ##       else predict class 0:  Yhat[i] = self.classes[0]
        return Yhat


    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopIter=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        ## First do some bookkeeping and setup:
        self.theta,X,Y = twod(self.theta), arr(X), arr(Y)   # convert to numpy arrays
        M,N = X.shape
        if Y.shape[0] != M:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        if self.theta.shape[1] != N+1:         # if self.theta is empty, initialize it!
            self.theta = np.random.randn(1,N+1)
        # Some useful modifications of the data matrices:
        X1  = np.hstack((np.ones((M,1)),X))    # make data array with constant feature
        Y01 = toIndex(Y, self.classes)         # convert Y to canonical "0 vs 1" classes

        it   = 0
        done = False
        Jsur = []
        J01  = []
        while not done:
            step = (2.0 * initStep) / (2.0 + it)   # common 1/iter step size change

            for i in range(M):  # for each data point i:
                ## TODO: compute zi = linear response of X[i,:]
                ## TODO: compute prediction yi 
                ## TODO: compute soft response si = logistic( zi )
                ## TODO: compute gradient of logistic loss wrt data point i:
                
                # Take a step down the gradient
                self.theta = self.theta - step * gradi

            # each pass, compute surrogate loss & error rates:
            J01.append( self.err(X,Y) )
            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = sum_i [ (si log si) if yi==1 else ((1-si)log(1-si)) ]
            Jsur.append( NotImplemented ) ## TODO ...
           
            ## For debugging: print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]  
            # raw_input()   # pause for keystroke

            # check stopping criteria:
            it += 1
            done = (it > stopIter) or ( (it>1) and (abs(Jsur[-1]-Jsur[-2])<stopTol) )


################################################################################
################################################################################
################################################################################

