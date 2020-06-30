import numpy as np
import scipy.spatial.distance as ssd
from scipy.stats import spearmanr

class UnivariateMetric(object):

    """
    Wrapper class to compute the similarity between two matrices.

    Here, we provide only Spearman rho and Pearson R correlation
    similarities.
    """

    def __init__(self, metric='spearman'):

        self.metric=metric
    
    def fit(self, X, y=None):

        function_map = {'spearman': spearman,
                        'pearson': pearson}
        
        return function_map[self.metric](X, y)


class SubspaceMetric(object):

    """
    Wrapper class to compute the distance between two equi-dimensional subspaces.
    For two subspaces, X and Y, we compute the inner product matrix, S,
    as transpose(X)xY.  

    Each distance measures is a unique function of the principle angles
    of S, which themselves are the inverse cosine of the singular values of S.
    """
    
    def __init__(self, metric='Grassmann'):

        self.metric = metric
    
    def fit(self, theta):

        function_map = {'Asimov': asimov,
                             'BinetCauchy': binetcauchy,
                             'Chordal': chordal,
                             'FubiniStudy': fubinistudy,
                             'Grassmann': grassmann,
                             'Martin': martin,
                             'Procrustes': procrustes,
                             'Projection': projection,
                             'Spectral': spectral}
        
        return function_map[self.metric](theta)


def grassmann(theta):
    
    """
    Compute Grassman distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """

    return np.sqrt((theta**2).sum())


def asimov(theta):
    
    """
    Compute Asimov distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """

    return theta[-1]


def binetcauchy(theta):
    
    """
    Compute Binet-Cauchy distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return np.sqrt((1 - np.prod(np.cos(theta)**2)))


def chordal(theta):
    
    """
    Compute Chordal distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return np.sqrt((np.sin(theta)**2).sum())


def fubinistudy(theta):
    
    """
    Compute Fubini-Study distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return np.arccos(np.prod(np.cos(theta)))


def martin(theta):
    
    """
    Compute Martin distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return np.sqrt(np.log(np.prod((1/(np.cos(theta)**2)))))


def procrustes(theta):
    
    """
    Compute Procrustes distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return 2*np.sqrt((np.sin(theta/2)**2).sum())


def projection(theta):
    
    """
    Compute Projection distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return np.sin(theta[-1])


def spectral(theta):
    
    """
    Compute Spectral distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return 2*np.sin(theta[-1]/2)


def spearman(X, y=None):

    """
    Compute the Spearman rho rank correlation of a matrix's features.

    Parameters:
    - - - - -
    X: float, array
        N by K matrix
    y: float, array
        optional
        N by M matrix
    
    Returns:
    - - - -
    rho: float, array
        (M+K) by (M+K)  correlation matrix
    """

    rho = spearmanr(X, y)[0]

    return rho

def pearson(X, y=None):

    """
    Compute the Pearson R correlation of a matrix's features.

    Parameters:
    - - - - -
    X: float, array
        N by K matrix
    y: float, array
        optional
        N by M matrix

    Returns:
    - - - -
    sim: float, array
        K by K correlation matrix
        K by M cross-correlation matrix (if y defined)
    """

    xd = X.ndim
    
    if not y:
        sim = ssd.squareform(ssd.pdist(X.T, metric='correlation'))
        sim = 1-sim

    if y:
        yd = y.ndim
        if yd == 1:
            y = y[None,:]
        
        sim = ssd.cdist(X.T, y, metric='correlation')
        sim = 1-sim
    
    return sim

        
