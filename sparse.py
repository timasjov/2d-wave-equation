"""
Classes and methods for sparse matrices.
"""

import common
import numpy as np
import fnum as f


class SparseMatrix:
    """Sparse matrix in triple storage format."""

    def __init__(self, m, n, nnz):
        """Sparse matrix
        @arg m: number of rows
        @arg n: number of columns
        @arg nnz: number of non-zero elements"""

        self.m = m
        self.n = n
        self.nnz = nnz
        self.irows = np.zeros(nnz, dtype=int)
        self.icols = np.zeros(nnz, dtype=int)
        self.vals = np.zeros(nnz, dtype=float)

    def __str__(self):
        """String representation of the matrix."""
        vs = []
        for i in xrange(self.nnz):
            vs.append("%d %d %.15f" % (self.irows[i], self.icols[i], self.vals[i]))
        vs_str = "\n".join(vs)
        return "%d x %d nnz=%d\n%s" % (self.m, self.n, self.nnz, vs_str)

    def __mul__(self, vec):
        """Matrix vector multiplication.
        @arg vec: NumPy vector
        """
        assert self.n == vec.size
        if common.OPTIMIZE_AX:

            mul_vector = vec[self.icols]
            mul_vector.shape = (1, len(self.vals))
            c = mul_vector * self.vals

            irows1 = ([self.irows[i] + 1 for i in range(len(self.irows))])
            y = f.nested.sum(c, irows1, self.n)
        else:
            y = np.zeros(vec.shape, dtype=float)

            for k in range(0, self.nnz):
                i = self.irows[k]
                j = self.icols[k]
                y[i] += vec[j] * self.vals[k]

        return y.reshape(vec.size, 1)

    def __getitem__(self, (i, j)):
        for k in range(self.nnz):
            if self.irows[k] == i and self.icols[k] == j:
                return self.vals[k]
        return 0.0


def as_sparse_matrix(matrix):
    """Create sparse matrix from NumPy matrix."""
    irows, icols = matrix.nonzero()
    vals = matrix[irows, icols]

    result = SparseMatrix(matrix.shape[0], matrix.shape[1], irows.size)
    result.irows[:] = irows
    result.icols[:] = icols
    result.vals[:] = vals
    return result