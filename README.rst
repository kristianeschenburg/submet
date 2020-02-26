======
submet
======

.. image:: https://img.shields.io/travis/kristianeschenburg/submet.svg
        :target: https://travis-ci.org/kristianeschenburg/submet

.. image:: https://img.shields.io/pypi/v/submet.svg
        :target: https://pypi.python.org/pypi/submet


Package to compute the distance between equi-dimensional subspaces.  Implements a variety of metrics.

Usage
--------
.. code-block:: python

        import numpy as np
        from submet import subspace

        X = np.random.rand(10, 5)
        Y = np.random.rand(10, 5)

        metric='grassmann'

        S = subspace.SubspaceDistance(metric)
        S.fit(X, Y)

        print('Distance: %.3f' % (S.distance_)

Features
---------

Given two subspaces, :math:`X` and :math:`Y`, each of dimension :math:`p`, the distance between the two subspaces is dependent on the singular values, :math:`\\Sigma` of their inner product.

Implemented metrics include:
        * Asimov
        * Binet-Cauchy
        * Chordal
        * Fubini-Study
        * Grassmann
        * Martin
        * Procrustes
        * Projection
        * Spectral