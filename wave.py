#! /usr/bin/env python
# -*- indent-tabs-mode: nil -*-
"""
Main script for wave simulation.
"""

import os
import sys
import getopt
import numpy as np
import math
import time

from images2gif import writeGif
from PIL import Image

import common
import cg
import sparse
import stationary

from matplotlib import pyplot as plt, animation, cm
from mpl_toolkits.mplot3d import Axes3D

#: name prefix for the temporary image files
TMP_PREFIX = "tmp2D_"
#: number of segments for (the each axis of) the grid
N = 70
#: number of time steps
NT = 30
#: end time for the simulation (start time is 0)
T = 3.0


def main():
    """Main function of wave simulation."""

    crankNicolson = False

    A_laplace = generateMatrix(common.STIFFNESS_FUNCTION, crankNicolson)
    A = sparse.as_sparse_matrix(A_laplace)

    v = np.zeros((A.n, 1))
    v_prev = np.zeros((A.n, 1))
    v_next = np.zeros((A.n, 1))

    x = np.zeros(shape=(N - 1, N - 1))
    y = x

    segments = np.linspace(0, 1, N + 1)
    xy_values = segments[1:-1]

    for yr in range(len(xy_values)):
        for xr in range(len(xy_values)):
            x[xr, yr] = xy_values[yr]
            y[xr, yr] = xy_values[xr]

    x.shape = ((N - 1) * (N - 1), 1)
    y.shape = ((N - 1) * (N - 1), 1)

    # Initialize v and v_prev
    v = I(x, y)
    v_prev = I(x, y)

    if common.STIFFNESS_FUNCTION is None:
        common.STIFFNESS_FUNCTION = constant_stiffness

    # create and set preconditioner
    if common.PRECONDITIONER_CLASS is not None:
        common.PRECONDITIONER = common.PRECONDITIONER_CLASS(A, n_iter=common.SGS_ITERATIONS)

    clean_files()

    # simulation iterations
    step(A,v_prev, v, v_next)

def usage():
    print """wave [-h|--help] [-N <num>] [-T <real>] [-S <num>] [-v] [--sgs] [-O <num>] [--variable]
\t-h|--help\t this message

\t-N <num>\t grid size (default 50)
\t-T <real>\t simulation time (default 3.0)
\t-S <num>\t number of time steps (default 300)
\t-v      \t visualize
\t--sgs=<n>\t use symmetric Gauss-Seidel as preconditioner with n of iterations
\t-O <num>\t 0 - no optimization, 1 - optimize SGS,
\t\t\t 2 - optimize Ax, 3 - optimize both
\t--variable\t use variable stiffness coefficient for material"""


def parse_options():
    global N, T, NT

    try:
        opts, args = getopt.getopt(sys.argv[1:], "N:T:S:hvO:", ["help", "sgs=", "variable"])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-N":
            N = int(a)
        elif o == "-T":
            T = float(a)
        elif o == "-S":
            NT = int(a)
        elif o == "-v":
            common.VISUALIZE = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()

        elif o in ("--sgs"):
            n_iter = int(a)
            common.PRECONDITIONER_CLASS = SGSPreconditioner
            common.SGS_ITERATIONS = n_iter

        elif o in ("-O"):
            a = int(a)
            if a == 0:
                pass
            elif a == 1:
                common.OPTIMIZE_SGS = True
            elif a == 2:
                common.OPTIMIZE_AX = True
            elif a == 3:
                common.OPTIMIZE_SGS = True
                common.OPTIMIZE_AX = True
            else:
                usage()
                raise "Invalid optimization value: %d" % a

        elif o in ("--variable"):
            common.STIFFNESS_FUNCTION = variable_stiffness

        else:
            assert False, "unhandled option"


def clean_files():
    """Clean temporary files for movie."""
    tmpfiles = filter(lambda n: n.startswith(TMP_PREFIX), os.listdir("."))
    print "Removing %d temporary files" % len(tmpfiles)
    for fname in tmpfiles:
        os.remove(fname)


def generateMatrix(stiffness_f=None, crankNicolson=False):
    """Generate Laplace-like matrix for the wave equation.

    @arg stiffness_f: function that returns stiffness coefficient for a point,
    see L{variable_stiffness} for example
    """
    if stiffness_f is None:
        stiffness_f = constant_stiffness

    b = 1.0 if not crankNicolson else 0.5
    m = (N - 1) * (N - 1) # number of unknowns
    M = np.zeros(shape=(m, m))

    n = N - 1
    for i in range(0, m):
        M[i, i] = 4.0 + ((DX * DX) / (DT * DT))

    for i in range(0, m - 1):
        if i % n != (n - 1):
            M[i + 1, i] = -1.0
            M[i, i + 1] = -1.0

    for i in range(n, m):
        M[i - n, i] = -1.0
        M[i, i - n] = -1.0
    return M


def step(A, v_prev, v, v_next, crankNicolson=False, stiffness_f=None):
    """
    One step of simulation: calculate L{v_next} from L{v} and L{v_prev}.

     1. create right-hand side I{b} from L{v} and L{v_prev}
     2. solve M{Ax=b} with CG
     3. write I{x} to L{v_next}

    @arg A: sparse matrix from FDM
    @type A: L{SparseMatrix}
    @arg v_prev: wave values one step before
    @arg v: current wave values
    @arg v_next: wave values for the next step to be calculated
    """

    counter = 1

    for i in range(NT):
        print 'Step: %s' % (i + 1)

        # Initialize b
        b = (DX * DX) / (DT / DT) * (2 * v - v_prev)

        # Solve Av = b
        v_next = cg.conjugate_gradient(A, b, prec=common.PRECONDITIONER)

        # Set pointers
        v_prev = v
        v = v_next

        if common.VISUALIZE:
            v_png = v.copy()
            v_png.shape = (N - 1, N - 1)
            space = np.linspace(0, 1, N + 1)
            xv, yv = np.meshgrid(space, space)

            values = np.zeros(shape=(N + 1, N + 1))

            for i in range(1, N):
                for j in range(1, N):
                    values[i, j] = v_png[i - 1, j - 1]

            fig = plt.figure()
            ax3 = fig.add_subplot(111, projection='3d')

            ax3.cla() # clear the axis
            ax3.scatter([0, 1], [0, 1], [-1, 1])
            ax3.plot_surface(xv, yv, values, rstride=1, cstride=1,
                             cmap=cm.jet, antialiased=True)
            levels = np.linspace(-1.0, 1.0, 20)
            ax3.contour(xv, yv, values, rstride=1, cstride=1,
                        zdir='z', offset=-1.0, cmap=cm.jet, levels=levels)
            plt.savefig('%s%d.png' % (TMP_PREFIX, counter))
        counter += 1


def I(x, y):
    """Function that returns initial wave values (initial conditions)."""
    l_and = np.logical_and
    return np.where(l_and(l_and(x > .4, x < .6),
                          l_and(y < .6, y > .4)),
                    1.0,
                    0.0)


class SGSPreconditioner:
    """Symmetric Gauss-Seidel preconditioner class.

    The objects of this class are callable which allows them to be used
    in place of functions."""

    def __init__(self, A, n_iter=3):
        """Object constructor ."""
        self.A = A
        self.n_iter = n_iter
        print "n_iter=", n_iter

    def __call__(self, r):
        """Function that does preconditioning when the object is called."""
        z = stationary.sym_gauss_seidel(self.A, r, n_iter=self.n_iter)
        return z


def constant_stiffness(x, y):
    """Constant coefficient S{lambda}(x,y)=1.0 for wave equation"""
    return 1.0


def variable_stiffness(x, y):
    """Variable coefficient S{lambda}(x,y) for wave equation"""
    if 0.4 < x < 0.7 and y < 0.5:
        return .1
    else:
        return 1.0


def variable_stiffness2(x, y):
    """Variable coefficient S{lambda}(x,y) for wave equation"""
    return x + y

# --- MAIN ---

parse_options()

#: time step S{Delta}t
DT = T / NT
#: space step S{Delta}x
DX = 1.0 / N

_start_time = time.time()

main()

file_names = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
images = [Image.open(fn) for fn in file_names]
filename = "animation.GIF"
writeGif(filename, images, duration=0.2)

print "Wall time: %.15f" % (time.time() - _start_time)
