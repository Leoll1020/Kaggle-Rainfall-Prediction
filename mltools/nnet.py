import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat



################################################################################
## NNETCLASSIFY ################################################################
################################################################################

def _add1(X):
    return np.hstack( (np.ones((X.shape[0],1)),X) )

class nnetClassify(classifier):
    """A simple neural network classifier

    Attributes:
      classes: list of class (target) identifiers for the classifier
      layers : list of layer sizes [N,S1,S2,...,C], where N = # of input features, S1 = # of hidden nodes 
               in layer 1, ... , and C = the number of classes, or 1 for a binary classifier
      weights: list of numpy arrays containing each layer's weights, size e.g. (S1,N), (S2,S1), etc.
  
    """

    def __init__(self, *args, **kwargs):
        """Constructor for nnetClassify (neural net classifier).

        Parameters: see the "train" function; calls "train" if arguments passed

        Properties:
          classes : list of identifiers for each class
          wts     : list of coefficients (weights) for each layer of the NN
          activation : function for layer activation function & derivative
        """
        self.classes = []
        self.wts = []
        #self.set_activation(activation.lower())
        #self.init_weights(sizes, init.lower(), X, Y)
        self.Sig = lambda Z: np.tanh(Z)       ## TODO: make flexible
        self.dSig= lambda Z: 1.0 - np.tanh(Z)**2    # (internal layers nonlinearity & derivative)
        #self.Sig0 = self.Sig
        #self.dSig0= self.dSig
        self.Sig0 = lambda Z: 1.0/(1.0 + np.exp(-Z))   # final layer nonlinearity & derivative
        self.dSig0= lambda Z: np.exp(-Z) / (1.0+np.exp(-Z))**2

        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"


    def __repr__(self):
        to_return = 'Multi-layer perceptron (neural network) classifier\nLayers [{}]'.format(self.get_layers())
        return to_return


    def __str__(self):
        to_return = 'Multi-layer perceptron (neural network) classifier\nLayers [{}]'.format(self.get_layers())
        return to_return

    def nLayers(self):
        return len(self.wts)

    @property
    def layers(self):
        """Return list of layer sizes, [N,H1,H2,...,C]

        N = # of input features
        Hi = # of hidden nodes in layer i
        C = # of output nodes (usually # of classes or 1)
        """
        if len(self.wts):
            layers = [self.wts[l].shape[1] for l in range(len(self.wts))]
            layers.append( self.wts[-1].shape[0] )
        else:
            layers = []
        return layers

    @layers.setter
    def layers(self, layers):
        raise NotImplementedError
    # adapt / change size of weight matrices (?)



## CORE METHODS ################################################################

    def predictSoft(self, X):
        """Make 'soft' (per-class confidence) predictions of the neural network on data X.

        Args:
          X : MxN numpy array containing M data points with N features each

        Returns:
          P : MxC numpy array of C class probabilities for each of the M data
        """
        X = arr(X)                               # convert to numpy if needed
        L = self.nLayers()                       # get number of layers
        Z = _add1(X)                             # initialize: input features + constant term

        for l in range(L - 1):                   # for all *except output* layer:
            Z = Z.dot( self.wts[l].T )        # compute linear response of next layer
            Z = _add1( self.Sig(Z) )               # apply activation function & add constant term

        Z = Z.dot( self.wts[L - 1].T )      # compute output layer linear response
        Z = self.Sig0(Z)                         # apply output layer activation function
        if Z.shape[1]==1: Z = np.hstack( (2.0*self.Sig0(0.0)-Z,Z) )  # if binary classifier, make Mx2 
        return Z


    def train(self, X, Y, init='zeros', stepsize=.01, stopTol=1e-4, stopIter=5000):
        """Train the neural network.

        Args:
          X : MxN numpy array containing M data points with N features each
          Y : Mx1 numpy array of targets (class labels) for each data point in X
          sizes : [Nin, Nh1, ... , Nout] 
              Nin is the number of features, Nout is the number of outputs, 
              which is the number of classes. Member weights are {W1, ... , WL-1},
              where W1 is Nh1 x Nin, etc.
          init : str 
              'none', 'zeros', or 'random'.  inits the neural net weights.
          stepsize : scalar
              The stepsize for gradient descent (decreases as 1 / iter).
          stopTol : scalar 
              Tolerance for stopping criterion.
          stopIter : int 
              The maximum number of steps before stopping. 
          activation : str 
              'logistic', 'htangent', or 'custom'. Sets the activation functions.
        
        """
        if self.wts[0].shape[1] - 1 != len(X[0]):
            raise ValueError('layer[0] must equal the number of columns of X (number of features)')

        self.classes = self.classes if len(self.classes) else np.unique(Y)

        if len(self.classes) != self.wts[-1].shape[0]: # and (self.wts[-1].shape[0]!=1 or len(self.classes)!=2):
            raise ValueError('layers[-1] must equal the number of classes in Y, or 1 for binary Y')


        M,N = mat(X).shape                          # d = dim of data, n = number of data points
        C = len(self.classes)                       # number of classes
        L = len(self.wts)                           # get number of layers

        Y_tr_k = to1ofK(Y,self.classes)             # convert Y to 1-of-K format

        # outer loop of stochastic gradient descent
        it = 1                                      # iteration number
        nextPrint = 1                               # next time to print info
        done = 0                                    # end of loop flag
        J01, Jsur = [],[]                           # misclassification rate & surrogate loss values

        while not done:
            step_i = float(stepsize) / it           # step size evolution; classic 1/t decrease
            
            # stochastic gradient update (one pass)
            for j in range(M):
                A,Z = self.__responses(twod(X[j,:]))  # compute all layers' responses, then backdrop
                delta = (Z[L] - Y_tr_k[j,:]) * arr(self.dSig0(Z[L]))  # take derivative of output layer

                for l in range(L - 1, -1, -1):
                    grad = delta.T.dot( Z[l] )      # compute gradient on current layer wts
                    delta = delta.dot(self.wts[l]) * arr(self.dSig(Z[l])) # propagate gradient down
                    delta = delta[:,1:]             # discard constant feature
                    self.wts[l] -= step_i * grad    # take gradient step on current layer wts

            J01.append(  self.err_k(X, Y_tr_k) )    # error rate (classification)
            Jsur.append( self.mse_k(X, Y_tr_k) )    # surrogate (mse on output)

            if it >= nextPrint:
                print('it {} : Jsur = {}, J01 = {}'.format(it,Jsur[-1],J01[-1]))
                nextPrint *= 2

            # check if finished
            done = (it > 1) and (np.abs(Jsur[-1] - Jsur[-2]) < stopTol) or it >= stopIter
            it += 1




    def err_k(self, X, Y):
        """Compute misclassification error rate. Assumes Y in 1-of-k form.  """
        return self.err(X, from1ofK(Y,self.classes).ravel())
        
        
    def mse(self, X, Y):
        """Compute mean squared error of predictor 'obj' on test data (X,Y).  """
        return mse_k(X, to1ofK(Y))


    def mse_k(self, X, Y):
        """Compute mean squared error of predictor; assumes Y is in 1-of-k format.  """
        return np.power(Y - self.predictSoft(X), 2).sum(1).mean(0)


## MUTATORS ####################################################################


    #def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
    def setActivation(self, method, sig=None, sig0=None): 
        """
        This method sets the activation functions. 

        Parameters
        ----------
        method : string, {'logistic' , 'htangent', 'custom'} -- which activation type
        Optional arguments for "custom" activation:
        sig : function object F(z) returns activation function & its derivative at z (as a tuple)
        sig0: activation function object F(z) for final layer of the nnet
        """
        raise NotImplementedError  # unfinished / tested
        method = method.lower()

        if method == 'logistic':
            self.sig = lambda z: twod(1 / (1 + np.exp(-z)))
            self.d_sig = lambda z: twod(np.multiply(self.sig(z), (1 - self.sig(z))))
            self.sig_0 = self.sig
            self.d_sig_0 = self.d_sig
        elif method == 'htangent':
            self.sig = lambda z: twod(np.tanh(z))
            self.d_sig = lambda z: twod(1 - np.power(np.tanh(z), 2))
            self.sig_0 = self.sig
            self.d_sig_0 = self.d_sig
        elif method == 'custom':
            self.sig = sig
            self.d_sig = d_sig
            self.sig_0 = sig_0
            self.d_sig_0 = d_sig_0
        else:
            raise ValueError('NNetClassify.set_activation: ' + str(method) + ' is not a valid option for method')

        self.activation = method



    def set_layers(self, sizes, init='random'):
        """
        Set layers sizes to sizes.

        Parameters
        ----------
        sizes : [int]
            List containing sizes.
        init : str (optional)
            Weight initialization method.
        """
        self.init_weights(sizes, init, None, None)


    def init_weights(self, sizes, init, X, Y):
        """
        This method sets layer sizes and initializes the weights of the neural network
          sizes = [Ninput, N1, N2, ... , Noutput], where Ninput = # of input features, and Nouput = # classes
          init = {'zeros', 'random'} : initialize to all zeros or small random values (breaks symmetry)
        """
        init = init.lower()

        if init == 'none':
            pass
        elif init == 'zeros':
            self.wts = [np.zeros((sizes[i + 1],sizes[i] + 1)) for i in range(len(sizes) - 1)]
        elif init == 'random':
            self.wts = [.0025 * np.random.randn(sizes[i+1],sizes[i]+1) for i in range(len(sizes) - 1)]
        else:
            raise ValueError('NNetClassify.init_weights: ' + str(init) + ' is not a valid option for init')



## HELPERS #####################################################################


    def __responses(self, Xin):
        """
        Helper function that gets linear sum from previous layer (A) and
        saturated activation responses (Z) for a data point. Used in:
            train
        """
        L = len(self.wts)
        A = [arr([1.0])]                                # initialize (layer 0)
        Z = [_add1(Xin)]                                # input to next layer: original features

        for l in range(1, L):
            A.append( Z[l - 1].dot(self.wts[l - 1].T) ) # linear response of previous later
            Z.append( _add1(self.Sig(A[l])) )           # apply activation & add constant feature

        A.append( Z[L - 1].dot(self.wts[L - 1].T) )     # linear response, output layer
        Z.append( self.Sig0(A[L]) )                     # apply activation (saturate for classifier, not regressor)

        return A,Z


################################################################################
################################################################################
################################################################################

class nnetRegress(regressor):
    """A simple neural network regressor

    Attributes:
      layers (list): layer sizes [N,S1,S2,...,C], where N = # of input features, 
                     S1 = # of hidden nodes in layer 1, ... , and C = the number of 
                     classes, or 1 for a binary classifier
      weights (list): list of numpy arrays containing each layer's weights, sizes 
                     (S1,N), (S2,S1), etc.
    """

    def __init__(self, *args, **kwargs):
        """Constructor for nnetRegress (neural net regressor).

        Parameters: see the "train" function; calls "train" if arguments passed

        Properties:
          wts     : list of coefficients (weights) for each layer of the NN
          activation : function for layer activation function & derivative
        """
        self.wts = [] 
        #self.set_activation(activation.lower())
        #self.init_weights(sizes, init.lower(), X, Y)
        self.Sig = lambda Z: np.tanh(Z)       ## TODO: make flexible
        self.dSig= lambda Z: 1.0 - np.tanh(Z)**2    # (internal layers nonlinearity & derivative)
        #self.Sig0 = self.Sig
        #self.dSig0= self.dSig
        self.Sig0 = lambda Z: Z               # final layer nonlinearity & derivative
        self.dSig0= lambda Z: 1.0+0*Z         #

        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"


    def __repr__(self):
        to_return = 'Multi-layer perceptron (neural network) regressor\nLayers [{}]'.format(self.get_layers())
        return to_return


    def __str__(self):
        to_return = 'Multi-layer perceptron (neural network) regressor\nLayers [{}]'.format(self.get_layers())
        return to_return

    def nLayers(self):
        return len(self.wts)

    @property
    def layers(self):
        """Return list of layer sizes, [N,H1,H2,...,C]
 
        N = # of input features
        Hi = # of hidden nodes in layer i
        C = # of output nodes (usually 1)
        """
        if len(self.wts):
            layers = [self.wts[l].shape[1] for l in range(len(self.wts))]
            layers.append( self.wts[-1].shape[0] )
        else:
            layers = []
        return layers

    @layers.setter
    def layers(self, layers):
        raise NotImplementedError
    # adapt / change size of weight matrices (?)



## CORE METHODS ################################################################

    def predict(self, X):
        """Make predictions of the neural network on data X.
        """
        X = arr(X)                          # convert to numpy if needed
        L = self.nLayers()                  # get number of layers
        Z = _add1(X)                        # initialize: input features + constant term

        for l in range(L - 1):              # for all *except output* layer:
            Z = Z.dot( self.wts[l].T )      # compute linear response of next layer
            Z = _add1( self.Sig(Z) )        # apply activation function & add constant term

        Z = Z.dot( self.wts[L - 1].T )      # compute output layer linear response
        Z = self.Sig0(Z)                    # apply output layer activation function
        return Z


    def train(self, X, Y, init='zeros', stepsize=.01, stopTol=1e-4, stopIter=5000):
        """Train the neural network.

        Args:
          X : MxN numpy array containing M data points with N features each
          Y : Mx1 numpy array of targets for each data point in X
          sizes (list of int): [Nin, Nh1, ... , Nout] 
              Nin is the number of features, Nout is the number of outputs, 
              which is the number of target dimensions (usually 1). Weights are {W1, ... , WL-1},
              where W1 is Nh1 x Nin, etc.
          init (str): 'none', 'zeros', or 'random'.  inits the neural net weights.
          stepsize (float): The stepsize for gradient descent (decreases as 1 / iter).
          stopTol (float): Tolerance for stopping criterion.
          stopIter (int): The maximum number of steps before stopping. 
          activation (str): 'logistic', 'htangent', or 'custom'. Sets the activation functions.
        """
        if self.wts[0].shape[1] - 1 != len(X[0]):
            raise ValueError('layer[0] must equal the number of columns of X (number of features)')

        if self.wts[-1].shape[0] > 1 and self.wts[-1].shape[0] != Y.shape[1]:
            raise ValueError('layers[-1] must equal the number of classes in Y, or 1 for binary Y')

        M,N = arr(X).shape                          # d = dim of data, n = number of data points
        L = len(self.wts)                           # get number of layers
        Y = arr(Y)
        Y2d = Y if len(Y.shape)>1 else Y[:,np.newaxis]

        # outer loop of stochastic gradient descent
        it = 1                                      # iteration number
        nextPrint = 1                               # next time to print info
        done = 0                                    # end of loop flag
        Jsur = []                                   # misclassification rate & surrogate loss values

        while not done:
            step_i = (2.0*stepsize) / (2.0+it)      # step size evolution; classic 1/t decrease
            
            # stochastic gradient update (one pass)
            for j in range(M):
                A,Z = self.__responses(twod(X[j,:]))  # compute all layers' responses, then backdrop
                delta = (Z[L] - Y2d[j,:]) * arr(self.dSig0(Z[L]))  # take derivative of output layer

                for l in range(L - 1, -1, -1):
                    grad = delta.T.dot( Z[l] )      # compute gradient on current layer wts
                    delta = delta.dot(self.wts[l]) * arr(self.dSig(Z[l])) # propagate gradient down
                    delta = delta[:,1:]             # discard constant feature
                    self.wts[l] -= step_i * grad    # take gradient step on current layer wts

            Jsur.append( self.mse(X, Y2d) )    # surrogate (mse on output)

            if it >= nextPrint:
                print('it {} : J = {}'.format(it,Jsur[-1]))
                nextPrint *= 2

            # check if finished
            done = (it > 1) and (np.abs(Jsur[-1] - Jsur[-2]) < stopTol) or it >= stopIter
            it += 1




## MUTATORS ####################################################################


    #def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
    def setActivation(self, method, sig=None, sig0=None): 
        """ This method sets the activation functions. 

        Args:
          method : string, {'logistic' , 'htangent', 'custom'} -- which activation type
        Optional arguments for "custom" activation:
          sig : f'n object F(z) returns activation function & its derivative at z (as a tuple)
          sig0: activation function object F(z) for final layer of the nnet
        """
        raise NotImplementedError  # unfinished / tested
        method = method.lower()

        if method == 'logistic':
            self.sig = lambda z: twod(1 / (1 + np.exp(-z)))
            self.d_sig = lambda z: twod(np.multiply(self.sig(z), (1 - self.sig(z))))
            self.sig_0 = self.sig
            self.d_sig_0 = self.d_sig
        elif method == 'htangent':
            self.sig = lambda z: twod(np.tanh(z))
            self.d_sig = lambda z: twod(1 - np.power(np.tanh(z), 2))
            self.sig_0 = self.sig
            self.d_sig_0 = self.d_sig
        elif method == 'custom':
            self.sig = sig
            self.d_sig = d_sig
            self.sig_0 = sig_0
            self.d_sig_0 = d_sig_0
        else:
            raise ValueError('nnetRegress.set_activation: ' + str(method) + ' is not a valid option for method')

        self.activation = method



    def set_layers(self, sizes, init='random'):
        """Set layers sizes to sizes.

        Args:
          sizes (int): List containing sizes.
          init (str, optional): Weight initialization method.
        """
        self.init_weights(sizes, init, None, None)


    def init_weights(self, sizes, init, X, Y):
        """Set layer sizes and initialize the weights of the neural network

        Args:
          sizes (list of int): [Nin, N1, N2, ... , Nout], where Nin = # of input features, and Nou = # classes
          init (str):  {'zeros', 'random'} initialize to all zeros or small random values (breaks symmetry)
        """
        init = init.lower()

        if init == 'none':
            pass
        elif init == 'zeros':
            self.wts = arr([np.zeros((sizes[i + 1],sizes[i] + 1)) for i in range(len(sizes) - 1)], dtype=object)
        elif init == 'random':
            self.wts = [.0025 * np.random.randn(sizes[i+1],sizes[i]+1) for i in range(len(sizes) - 1)]
        else:
            raise ValueError('nnetRegress.init_weights: ' + str(init) + ' is not a valid option for init')



## HELPERS #####################################################################


    def __responses(self, Xin):
        """
        Helper function that gets linear sum from previous layer (A) and
        saturated activation responses (Z) for a data point. Used in:
            train
        """
        L = len(self.wts)
        A = [arr([1.0])]                                # initialize (layer 0)
        Z = [_add1(Xin)]                                # input to next layer: original features

        for l in range(1, L):
            A.append( Z[l - 1].dot(self.wts[l - 1].T) ) # linear response of previous later
            Z.append( _add1(self.Sig(A[l])) )           # apply activation & add constant feature

        A.append( Z[L - 1].dot(self.wts[L - 1].T) )     # linear response, output layer
        Z.append( self.Sig0(A[L]) )                     # apply activation (saturate for classifier, not regressor)

        return A,Z


################################################################################
################################################################################
################################################################################

