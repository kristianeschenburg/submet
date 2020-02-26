import numpy as np

from metrics import Metric
class SubspaceDistance(object):
    
    """
    Class to compute a variety of distance between subspaces of equal dimension.
    """
    
    def __init__(self, metric='Grassmann'):
        
        """
        Parameters:
        - - - - -
        metric: string
            metric / distance to use
        """

        assert metric in ['Asimov', 'BinetCauchy', 'Chordal', 'FubiniStudy', 
                          'Grassmann', 'Martin', 'Procrustes', 'Projection',
                          'Spectral']
        
        self.metric = metric


    def fit(self, X, Y):
        
        """
        Fit subspace distance between two equi-dimensional subspaces.
        
        Parameters:
        - - - - -
        X, Y: float, array
            two equi-dimensional subspaces
        """

        M = Metric(self.metric)
        
        p = np.min([X.shape[1], Y.shape[1]])
        [q1, r1] = np.linalg.qr(X)
        [q2, r2] = np.linalg.qr(Y)
        
        S = q1.T.dot(q2)
        [u, s, v] = np.linalg.svd(S, full_matrices=False)
        
        theta = np.arccos(s)
        
        self.x_ = q1.dot(u)
        self.y_ = q2.dot(v)
        
        print('Computing %s distance.' % (self.metric))
        self.distance_ = M.fit(theta)