"""Stationary iterative methods for solving sparse linear systems,
like Gauss-Seidel.

The accepted matrix object is of L{SparseMatrix} class.
"""

import common
import numpy as np
import fnum as f


def sym_gauss_seidel(A, b, n_iter):
    """Applies L{n_iter} iterations of Symmetric Gauss-Seidel method
    to a zero vector and returns the result.

    if L{common.OPTIMIZE_SGS} is True then Fortran code of SGS is used
    instead of the Python one.

    @arg A: I{m} times I{n} sparse matrix
    @type A: L{SparseMatrix}
    @arg b: right-hand side vector
    @type b: C{numpy.ndarray} of size I{m}
    @return: the resulting vector of L{n_iter} SGS iterations
    @rtype: C{numpy.ndarray} of shape (I{n},1)
    """

    k = n_iter
    z = np.zeros(len(b))
    n = len(b)

    if common.OPTIMIZE_SGS:
        A.irows1 = ([A.irows[i] + 1 for i in range(len(A.irows))])
        A.icols1 = ([A.icols[i] + 1 for i in range(len(A.icols))])
        z = f.stationary.sym_gauss_seidel(A.irows1, A.icols1, A.vals, b, n_iter)

    else:

        for k in range(k):
            for i in range(n):

                sum = 0.0
                for j in range(n):
                    if i != j:
                        sum += A[i, j] * z[j]
                z[i] = 1.0 / A[i, i] * (b[i] - sum)

            for i in range(n - 1, -1, -1):

                sum = 0.0
                for j in range(n):
                    if i != j:
                        sum += A[i, j] * z[j]
                z[i] = 1.0 / A[i, i] * (b[i] - sum)

    return z.reshape((z.size, 1))