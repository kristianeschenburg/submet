class Metric(object):

    """
    Wrapper class to compute the distance between two equi-dimensional subspaces.
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
        
        return function_map[metric](theta)


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
    Compute Projection distance between two equi-dimensional subspaces.
    
    Parameters:
    - - - - -
    theta: float, array
        arccos(singular values) of inner product of two equi-dimensional subspaces
    """
    
    return 2*np.sin(theta[-1]/2)