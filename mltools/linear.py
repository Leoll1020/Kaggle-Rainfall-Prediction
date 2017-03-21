
import math
import numpy as np
from functools import reduce

from numpy import asmatrix as mat
from numpy import asarray as arr

from .base import regressor


################################################################################
## Linear models ###############################################################
################################################################################


class linearRegress(regressor):

	def __init__(self, *args, **kwargs):
		"""
		Constructor for a linear regression model

		Parameters
		----------
		X : M x N numpy array that contains M data points of N features.
		Y : M x 1 numpy array of target values associated with each data point in X
		reg : scalar (int or float) 
			L2 regularization penalty: minimize  (1/M) ||y - X*w'||^2 + reg * ||w||^2.
		"""
		self.theta = []

		if len(args) or len(kwargs):
			self.train(*args,**kwargs)


	def __repr__(self):
		str_rep = 'linearRegress model, {} features\n{}'.format(
			len(self.theta), self.theta)
		return str_rep


	def __str__(self):
		str_rep = 'linearRegress model, {} features\n{}'.format(
			len(self.theta), self.theta)
		return str_rep


	def train(self, X, Y, reg=0):
		"""
		This method trains a linear regression predictor on the given data.
		See the constructor doc string for arguments.
		"""
		X,Y = mat(X), mat(Y)		# force matrix types (for linear algebra)
		M,N = X.shape
		if Y.shape[0] != M:
			Y = Y.T               # try transposing if needed
		if Y.shape[0] != M:     # if that doesn't work, give up
			raise ValueError('X and Y must have the same number of data points')

		X_train = np.concatenate((np.ones((M,1)), X), axis=1)		# extend features by including a constant feature

		if reg == 0:
			self.theta = np.linalg.lstsq(X_train, Y, rcond=0.0)[0].T	# solve least-squares via numpy: Th = Y/X
		else:
			m,n = mat(X_train).shape                            # or solve manually via pseudo-inverse
			self.theta = (Y.T * X_train / m) * np.linalg.inv(X_train.T * (X_train / m) + reg * np.eye(n))

		self.theta = arr(self.theta)                          # theta should be a row-vector (array)
		#self.theta = arr(self.theta).ravel()									# make sure self.theta is flat


	def predict(self, X):
		"""
		Predict: Yi = Xi * theta

		Parameters
		----------
		X : M x N numpy array that contains M data points with N features.
		"""
		return self.theta[:,0].T + X.dot(self.theta[:,1:].T)
		#return self.theta[0] + X.dot(self.theta[1:,np.newaxis])




