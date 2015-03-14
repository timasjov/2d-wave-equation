"""
Conjugate Gradient method with preconditioning for solving sparse
linear systems.

@var TOLERANCE: iterate in CG until the error is less than this value
@var MAX_ITER: maximum number of CG iterations
"""

import numpy as np
import sparse
import time

TOLERANCE = 1E-10
MAX_ITER = 1000


def conjugate_gradient(A, b, prec=None):
    """Solve the system of linear equations using (preconditioned)
    Conjugate Gradient method.

    @arg A: I{m} times I{n} sparse matrix
    @type A: L{SparseMatrix}
    @arg b: right-hand side vector
    @type b: C{numpy.ndarray} of shape (I{m},1)
    @return: solution vector
    @rtype: C{numpy.ndarray} of shape (I{n},1)
    @arg prec: preconditioner function (or any callable) that returns I{z} for given I{r}
    @type prec: 1-argument Python callable
    """

    t1 = time.clock()
    x = np.zeros((A.n, 1))
    r = b - A * x
    it = 0

    while np.sqrt(np.sum(r ** 2)) > TOLERANCE and it < MAX_ITER:

        if prec:
            z = prec(r)
        else:
            z = r.copy()

        rt = r.transpose()
        rtr = 0
        for j in range(len(r)):
            rtr += rt[0, j] * z[j, 0]

        if it == 0:
            p = z
        else:
            beta = rtr / rtr_last
            p = z + beta * p_last

        Ap = A * p
        pt = p.transpose()
        ptAp = 0

        for j in range(len(Ap)):
            ptAp += pt[0, j] * Ap[j, 0]

        alfa = rtr / ptAp
        x_next = x + alfa * p
        r_next = r - alfa * Ap

        x = x_next.copy()
        r = r_next.copy()

        rtr_last = rtr.copy()
        p_last = p.copy()
        it += 1

    t2 = time.clock()
    print "CG number of iterations: %d" % it
    print "CG error: %.15f" % np.sqrt(np.sum(r ** 2))
    print "CG time: %.15f" % (t2 - t1)

    return x