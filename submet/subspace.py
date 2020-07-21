from numba import prange
import numpy as np

from submet.metrics import SubspaceMetric


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

    def fit(self, X, Y=None, axis=0):
        """
        Fit subspace distance between two equi-dimensional subspaces.

        Parameters:
        - - - - -
        X, Y: float, array
            two equi-dimensional subspaces
            Y is optional

        If Y is not provided, output is an Sx by Sx matrix, where Sx is 
        the number of samples in X.
        If Y is provided, output is an Sx by Sy matrix, where Sx/Sy are 
        the number of samples in each matrix.

        """

        M = SubspaceMetric(self.metric)
        n = X.shape[axis]
        p = Y.shape[axis] if np.any(Y) else X.shape[axis]

        d = np.zeros((n, p))

        for i in prange(n):
            for j in prange(p):

                theta = self._pabs(X[i], Y[j] if np.any(Y) else X[j])
                td = M.fit(theta)
                d[i, j] = td

        if not np.any(Y):
            d[np.diag_indices(n)] = 0

        self.distance_ = d

    def _pabs(self, x, y):
        """
        Compute the principle angles between two subspaces.

        Parameters:
        - - - - -
        x,y: float, array
            two subspaces of each dimensions
        
        Returns:
        - - - -
        theta: float, array
            principle angles between the subspaces spanned 
            by the columns of x and y
        """

        if x.ndim == 1:
            x = x[:, None]

        if y.ndim == 1:
            y = y[:, None]

        # orthogonalize each subspace
        q1, _ = np.linalg.qr(x)
        q2, _ = np.linalg.qr(y)

        # compute inner product matrix
        S = q1.T.dot(q2)[None, :]
        [u, s, v] = np.linalg.svd(S, full_matrices=False)

        # compute principle angles
        theta = np.arccos(s)

        return theta
