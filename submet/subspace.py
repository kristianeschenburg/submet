import numpy as np
from submet.metrics import Metric


class SubspaceDistance(object):

    """
    Class to compute a variety of equi-dimensional subspace distances.
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

        # return dimensions of each subspace
        if X.ndim == 1:
            X = X[:, None]

        if Y.ndim == 1:
            Y = Y[:, None]

        # orthogonalize each subspace
        q1,_ = np.linalg.qr(X)
        q2,_ = np.linalg.qr(Y)

        # compute inner product matrix
        S = q1.T.dot(q2)[None, :]
        [u, s, v] = np.linalg.svd(S, full_matrices=False)

        # compute principle angles
        theta = np.arccos(s).squeeze()

        # compute subspace distance
        self.distance_ = M.fit(theta)
        self.x_ = q1.dot(u).squeeze()
        self.y_ = q2.dot(v).squeeze()
