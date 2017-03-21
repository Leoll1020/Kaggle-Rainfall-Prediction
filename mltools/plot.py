import numpy as np
import matplotlib.pyplot as plt

from numpy import atleast_2d as twod


################################################################################
##  PLOTTING FUNCTIONS #########################################################
################################################################################


def plotClassify2D(learner, X, Y, pre=lambda x: x, axis=None, nGrid=128, **kwargs):
    """
    Plot data and classifier outputs on two-dimensional data.
    This function plot data (X,Y) and learner.predict(X, Y) 
    together. The learner is is predicted on a dense grid
    covering data X, to show its decision boundary.

    Parameters
    ----------
    learner : learner object
        A trained learner object that inherits from one of
        the 'Classify' or 'Regressor' base classes.
    X : numpy array
        N x M array of data; N = number of data, M = dimension
        (number of features) of data.
    Y : numpy array
        1 x N arra containing labels corresponding to data points
        in X.
    pre : function object (optional)
        Function that is applied to X before prediction.
    axis  : a matplotlib axis / plottable object (optional)
    nGrid : density of 2D grid points (default 128)
    """

    if twod(X).shape[1] != 2:
        raise ValueError('plotClassify2D: function can only be called using two-dimensional data (features)')

    # TODO: Clean up code

    if axis == None: axis = plt 
    hld = axis.ishold();
    axis.hold(True);
    axis.plot( X[:,0],X[:,1], 'k.', visible=False )
    # TODO: can probably replace with final dot plot and use transparency for image (?)
    ax = axis.axis()
    xticks = np.linspace(ax[0],ax[1],nGrid)
    yticks = np.linspace(ax[2],ax[3],nGrid)
    grid = np.meshgrid( xticks, yticks )

    XGrid = np.column_stack( (grid[0].flatten(), grid[1].flatten()) )
    if learner is not None:
        YGrid = learner.predict( pre(XGrid) )
        #axis.contourf( xticks,yticks,YGrid.reshape( (len(xticks),len(yticks)) ), nClasses )
        axis.imshow( YGrid.reshape( (len(xticks),len(yticks)) ), extent=ax, interpolation='nearest',origin='lower',alpha=0.5, aspect='auto' )
    cmap = plt.cm.get_cmap()
    # TODO: if Soft: predictSoft; get colors for each class from cmap; blend pred with colors & show
    #  
    try: classes = np.array(learner.classes);
    except Exception: classes = np.unique(Y)
    cvals = (classes - min(classes))/(max(classes)-min(classes)+1e-100)
    for i,c in enumerate(classes): 
        axis.plot( X[Y==c,0],X[Y==c,1], 'ko', color=cmap(cvals[i]), **kwargs )  
    axis.axis(ax); axis.hold(hld)


def histy(X,Y,axis=None,**kwargs):
    """
    Plot a histogram (using matplotlib.hist) with multiple classes of data
    Any additional arguments are passed directly into hist()
    Each class of data are plotted as a different color
    To specify specific histogram colors, use e.g. facecolor={0:'blue',1:'green',...}
      so that facecolor[c] is the color for class c
    Related but slightly different appearance to e.g.
      matplotlib.hist( [X[Y==c] for c in np.unique(Y)] , histtype='barstacked' )
    """
    if axis == None: axis = plt 
    yvals = np.unique(Y)
    nil, bin_edges = np.histogram(X, **kwargs)
    C,H = len(yvals),len(nil)
    hist = np.zeros( shape=(C,H) )
    cmap = plt.cm.get_cmap()
    cvals = (yvals - min(yvals))/(max(yvals)-min(yvals)+1e-100)
    widthFrac = .25+.75/(1.2+2*np.log10(len(yvals)))
    for i,c in enumerate(yvals):
        histc,nil = np.histogram(X[Y==c],bins=bin_edges)
        hist[i,:] = histc
    for j in xrange(H):
        for i in np.argsort(hist[:,j])[::-1]:
            delta = bin_edges[j+1]-bin_edges[j]
            axis.bar(bin_edges[j]+delta/2*i/C*widthFrac,hist[i,j],width=delta*widthFrac,color=cmap(cvals[i]))



def plotPairs(X,Y=None,**kwargs):
    """
    Plot all pairs of features in a grid
    Diagonal entries are histograms of each feature
    Off-diagonal are 2D scatterplots of pairs of features
    """
    m,n = X.shape
    if Y is None: Y = np.ones( (m,) )
    fig,ax = plt.subplots(n,n)
    for i in range(n):
        for j in range(n):
            if i == j:
                histy(X[:,i],Y,axis=ax[j,i])
            else:
                plot_classify_2D(None,X[:,[i,j]],Y,axis=ax[j,i])
            

def plotGauss2D(mu,cov,*args,**kwargs):
    """
    Plot an ellipsoid indicating (one std deviation of) a 2D Gaussian distribution
    All additional arguments are passed into plot(.)
    """
    from scipy.linalg import sqrtm
    theta = np.linspace(0,2*np.pi,50)
    circle = np.array([np.sin(theta),np.cos(theta)])
    ell = sqrtm(cov).dot(circle)
    ell += twod(mu).T

    plt.plot( mu[0],mu[1], 'x', ell[0,:],ell[1,:], **kwargs)



# TODO: plotSoftClassify2D



# TODO: plotRegress1D




################################################################################
################################################################################
################################################################################
