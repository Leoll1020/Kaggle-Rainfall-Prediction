import csv
import math
import numpy as np
import random

from .base import classifier
from numpy import asarray as arr


# Bayes classifiers

# Current implementation only includes a Gaussian class-conditional model,
# "gaussClassify"

# TODO: data weights

################################################################################
## GAUSSBAYESCLASSIFY ##########################################################
################################################################################


class gaussClassify(classifier):
    """Bayes Classifier with Gaussian class-conditional probabilities.  """

    def __init__(self, X=None, Y=None, equal=0, diagonal=0, wts=None, reg=0):
        """Constructor for a Gaussian Bayes Classifier. 

        Args:
          X (array): M,N array of M data points with N features each
          Y (vector): M, or M,1 array of the targets (class labels) for each data point
          equal (bool): Force all classes to share a single covariance model
          diagonal (bool): Force all classes to use diagonal covariance models
          wts (vector): M, or M,1 array of positive weights (floats)
          reg (float): L2 regularization term for the covariance estimator

        Properties:
          classes (list):  list of class identifiers
          probs   (list):  list of class probabilities for each class
          means   (list):  list of numpy arrays (1xN); mean of each class distribution
          covars  (list):  list of numpy arrays (NxN); covariances of each class distribution
        """
        self.means = []
        self.covars = []
        self.probs = []
        self.classes = []

        if type(X) == np.ndarray and type(Y) == np.ndarray:
            self.train(X, Y, equal, diagonal, wts, reg)


    def __repr__(self):
        to_print = 'Gaussian classifier, {} classes:\n{}\nMeans:\n{}\nCovariances:\n{}\n'.format(
            len(self.classes), self.classes, 
            str([str(np.asmatrix(m).shape[0]) + ' x ' + str(np.asmatrix(m).shape[1]) for m in self.means]), 
            str([str(np.asmatrix(c).shape[0]) + ' x ' + str(np.asmatrix(c).shape[1]) for c in self.covars])) 
        return to_print

    
    def __str__(self):
        to_print = 'Gaussian classifier, {} classes:\n{}\nMeans:\n{}\nCovariances:\n{}\n'.format(
            len(self.classes), self.classes, 
            str([str(np.asmatrix(m).shape[0]) + ' x ' + str(np.asmatrix(m).shape[1]) for m in self.means]), 
            str([str(np.asmatrix(c).shape[0]) + ' x ' + str(np.asmatrix(c).shape[1]) for c in self.covars])) 
        return to_print


## CORE METHODS ################################################################


    def train(self, X, Y, equal=0, diagonal=0, wts=None, reg=0):
        """Train the model on data (X,Y).

        This method trains a Bayes classifier with class models. Refer to 
        the constructor doc string for descriptions of X and Y.
        """
        M,N = X.shape
        wts = wts if type(wts) == np.ndarray else [1.0 for i in range(len(Y))]
        wts = np.divide(wts, np.sum(wts))

        # get classes if needed
        self.classes = list(np.unique(Y)) # if type(Y) == np.ndarray else []
        self.probs  = [0.0 for c in self.classes]
        self.means  = [np.zeros((1,N)) for c in self.classes]
        self.covars = [np.zeros((N,N)) for c in self.classes]

        for i,c in enumerate(self.classes):
            indexes = np.where(Y == c)[0] 
            self.probs[i] = np.sum(wts[indexes])                # compute the (weighted) fraction of data in class i

            wtsi = wts[indexes] / self.probs[i]                 # compute relative weights of data in this class
            self.means[i] = wtsi.T.dot( X[indexes,:] )          # compute the (weighted) mean

            X0 = X[indexes,:] - self.means[i]                   # center the data
            wX0 = X0 * wtsi[:,np.newaxis]                       # weighted, centered data
            if diagonal:            # brute-force weighted variance computation
                self.covars[i] = np.diag(X0.T.dot(wX0) + reg)
            else:                   # weighted, regularized covariance computation
                self.covars[i] = X0.T.dot(wX0) + np.diag(reg + 0 * self.means[i])

        if equal:                                               # force covariances to be equal (take weighted average)
            Cov = sum( [self.probs[i]*self.covars[i] for i in range(len(self.probs))] )
            for i,c in enumerate(self.classes):
              self.covars[i] = Cov


    def predictSoft(self, X):
        """Compute the posterior probabilities of each class for each datum in X

        Args:
            X (array): M,N array of M data points of N features to predict with

        Returns:
            P (array): M,C array of C class probabilities for each data point
        """
        m,n = X.shape
        C = len(self.classes)
        p = np.zeros((m, C))
        for c in range(C):                        # compute probabilities for each class by Bayes rule
            # p(c) * p(x|c)
            p[:,c] = self.probs[c] * self.__eval_gaussian(X, self.means[c], self.covars[c])
        p /= np.sum(p, axis=1, keepdims=True)     # normalize each row (data point)
        return p


#    def predict(self, X):
#        """ Predict the class value of each data point in X """
#        Not implemented -- use default (predictSoft then find most likely class)



## HELPERS #####################################################################

    def __eval_gaussian(self, X, mean, covar):
        """A helper method that calculates the probability of X under a Gaussian distribution.  """
        m,d = X.shape
        p = np.zeros((m, 1))                                    # store evaluated probabilities for each datum
        R = X - np.tile(mean, (m, 1))                          # compute probability of Gaussian at each point
        # need inverse covariance and normalizing constant for Gaussian
        if len(covar.shape) > 1:         # use matrix inverse,
          constant = 1 / (2 * math.pi)**(d / 2) / np.linalg.det(covar)**(0.5)
          inverse = np.linalg.inv(covar) 
          p = np.exp(-0.5 * np.sum(np.dot(R, inverse) * R, axis=1)) * constant
        else:                            # or simple inverse for diagonal matrices
          constant = 1 / (2 * math.pi)**(d / 2) / np.prod(covar)**(0.5)
          inverse = 1.0 / covar[np.newaxis,:]
          p = np.exp(-0.5 * np.sum((R*inverse) * R, axis=1)) * constant
        # (vectorized)
        return p
            
        
