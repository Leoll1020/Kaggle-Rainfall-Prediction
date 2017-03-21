import numpy as np

from numpy import loadtxt as loadtxt
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from scipy.linalg import sqrtm



################################################################################
## Methods for creating / sampling synthetic datasets ##########################
################################################################################


def data_gauss(N0, N1=None, mu0=arr([0, 0]), mu1=arr([1, 1]), sig0=np.eye(2), sig1=np.eye(2)):
	"""Sample data from a two-component Gaussian mixture model.  	

	Args:
	    N0 (int): Number of data to sample for class -1.
	    N1 :(int) Number of data to sample for class 1.
	    mu0 (arr): numpy array
	    mu1 (arr): numpy array
	    sig0 (arr): numpy array
	    sig1 (arr): numpy array

	Returns:
	    X (array): Array of sampled data
	    Y (array): Array of class values that correspond to the data points in X.

	TODO: test more
	"""
	if not N1:
		N1 = N0

	d1,d2 = twod(mu0).shape[1],twod(mu1).shape[1]
	if d1 != d2 or np.any(twod(sig0).shape != arr([d1, d1])) or np.any(twod(sig1).shape != arr([d1, d1])):
		raise ValueError('data_gauss: dimensions should agree')

	X0 = np.dot(np.random.randn(N0, d1), sqrtm(sig0))
	X0 += np.ones((N0,1)) * mu0
	Y0 = -np.ones(N0)

	X1 = np.dot(np.random.randn(N1, d1), sqrtm(sig1))
	X1 += np.ones((N1,1)) * mu1
	Y1 = np.ones(N1)

	X = np.row_stack((X0,X1))
	Y = np.concatenate((Y0,Y1))

	return X,Y




def data_GMM(N, C, D=2, get_Z=False):
	"""Sample data from a Gaussian mixture model.  

  Builds a random GMM with C components and draws M data x^{(i)} from a mixture
	of Gaussians in D dimensions

	Args:
	    N (int): Number of data to be drawn from a mixture of Gaussians.
	    C (int): Number of clusters.
	    D (int): Number of dimensions.
	    get_Z (bool): If True, returns a an array indicating the cluster from which each 
		    data point was drawn.

	Returns:
	    X (arr): N x D array of data.
	    Z (arr): 1 x N array of cluster ids; returned also only if get_Z=True
    
	TODO: test more; N vs M
	"""
	C += 1
	pi = np.zeros(C)
	for c in range(C):
		pi[c] = gamrand(10, 0.5)
	pi = pi / np.sum(pi)
	cpi = np.cumsum(pi)

	rho = np.random.rand(D, D)
	rho = rho + twod(rho).T
	rho = rho + D * np.eye(D)
	rho = sqrtm(rho)
	
	mu = mat(np.random.randn(c, D)) * mat(rho)

	ccov = []
	for i in range(C):
		tmp = np.random.rand(D, D)
		tmp = tmp + tmp.T
		tmp = 0.5 * (tmp + D * np.eye(D))
		ccov.append(sqrtm(tmp))

	p = np.random.rand(N)
	Z = np.ones(N)

	for c in range(C - 1):
		Z[p > cpi[c]] = c
	Z = Z.astype(int)

	X = mu[Z,:]

	for c in range(C):
		X[Z == c,:] = X[Z == c,:] + mat(np.random.randn(np.sum(Z == c), D)) * mat(ccov[c])

	if get_Z:
		return (arr(X),Z)
	else:
		return arr(X)


def gamrand(alpha, lmbda):
	"""Gamma(alpha, lmbda) generator using the Marsaglia and Tsang method

	Args:
	    alpha (float): scalar
	    lambda (float): scalar
	
	Returns:
	    (float) : scalar

	TODO: test more
	"""
  # (algorithm 4.33).
	if alpha > 1:
		d = alpha - 1 / 3
		c = 1 / np.sqrt(9 * d)
		flag = 1

		while flag:
			Z = np.random.randn()	

			if Z > -1 / c:
				V = (1 + c * Z)**3
				U = np.random.rand()
				flag = np.log(U) > (0.5 * Z**2 + d - d * V + d * np.log(V))

		return d * V / lmbda

	else:
		x = gamrand(alpha + 1, lmbda)
		return x * np.random.rand()**(1 / alpha)



def data_mouse():
	"""Simple by-hand data generation using the GUI

	Opens a matplotlib plot window, and allows the user to specify points with the mouse.
	Each button is its own class (1,2,3); close the window when done creating data.

  Returns:
      X (arr): Mx2 array of data locations
      Y (arr): Mx1 array of labels (buttons)
	"""
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(-1,2), ylim=(-1,2))
	X  = np.zeros( (0,2) )
	Y  = np.zeros( (0,) )
	col = ['bs','gx','ro']
	
	def on_click(event):
		X.resize( (X.shape[0]+1,X.shape[1]) )
		X[-1,:] = [event.xdata,event.ydata]
		Y.resize( (Y.shape[0]+1,) )
		Y[-1] = event.button
		ax.plot( event.xdata, event.ydata, col[event.button-1])
		fig.canvas.draw()

	fig.canvas.mpl_connect('button_press_event',on_click)
        inter=plt.isinteractive(); hld=plt.ishold();
        plt.ioff(); plt.hold(True); plt.show();
        if inter: plt.ion();
        if not hld: plt.hold(False);
	return X,Y

