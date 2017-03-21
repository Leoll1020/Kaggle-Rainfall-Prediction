import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta )!= 3: raise ValueError('Data & model must be 2D');
        ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
        ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
        x1b = np.array([ax[0],ax[1]]);  # at X1 = points in x1b
        ##x2b = NotImplementedError;      # TODO find x2 values as a function of x1's values
        x2b=(-self.theta[0]-self.theta[1]*x1b)/self.theta[2]
        ## Now plot the data and the resulting boundary:
        ##A = Y==1;                                              # and plot it:
        A=Y==self.classes[0];
        plt.plot(X[A,0],X[A,1],'b.',X[-A,0],X[-A,1],'r.',x1b,x2b,'k-'); plt.axis(ax); plt.show();
    

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
        ##raise NotImplementedError
        ## TODO: compute linear response r[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] + ... for each i
        ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
        ##       else predict class 0:  Yhat[i] = self.classes[0]
        Yhat=np.array([])
        ##r=list()
        for i in range(len(X)):
            ##print(i)
            ##print(self.theta[0] + self.theta[1]*X[i,0] + self.theta[2]*X[i,1])
            tmp = self.theta[0] + self.theta[1]*X[i,0] + self.theta[2]*X[i,1]
            ##print(tmp)
            if (tmp >0):
                Yhat=np.concatenate((Yhat,np.array([self.classes[1]])))
                ##Yhat.append([self.classes[1]])
            else:
                Yhat=np.concatenate((Yhat,np.array([self.classes[0]])))
                ##Yhat.append([self.classes[0]])               
        return Yhat

    def sigma(self, z):
	return (1/ (1+ np.exp(-z)))

    def loss(self,z, y):
	sigma_resp = self.sigma(z)
	return -y * np.log(sigma_resp) - (1 - y) * np.log(1 - sigma_resp)

    def deri_sigma(self, z):
        ze = np.exp(-z)
        return self.sig(z)*(1-self.sig(z))

    def gradient(self, z ,x, y):  #For one data point
        sx = self.sigma(z)
        return -(y*(1-sx)*x-1+y)



    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X))   # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            surro_loss=0
            for i in np.random.permutation(M):
                ri    = XX[i,0];     # TODO: compute linear response r(x)
                gradi = self.gradient(ri,XX[i],YY[i]);     # TODO: compute gradient of NLL loss
                surro_loss+=self.loss(ri,YY[i]);
                self.theta -= stepsize * gradi;  # take a gradient step
            J01.append( self.err(X,Y) )  # evaluate the current error rate 

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = sum_i [ (log si) if yi==1 else (log(1-si)) ]
            Jnll.append(surro_loss/M) # TODO evaluate the current NLL loss
            plt.figure(1); plt.plot(Jnll,'b-',J01,'r-'); #plt.draw();    # plot losses
            if N==2: plt.figure(2); self.plotBoundary(X,Y); #plt.draw(); # & predictor if 2D
            plt.pause(.01);                    # let OS draw the plot

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]  
            # raw_input()   # pause for keystroke

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            #done = NotImplementedError;   # or if Jnll not changing between epochs ( < stopTol )
            done=(epoch > stopEpochs) or ((epoch>1) and (abs(Jnll[-1]-Jnll[-2])<stopTol));
            plt.plot(list(range(epoch)),Jnll,'r-',list(range(epoch)),J01,'b-');
        plt.draw();

    


################################################################################
################################################################################
################################################################################

