
"""Constants and variables common to the whole program

@var VISUALIZE: whether to generate video
@var PRECONDITIONER_CLASS: class or function that returns preconditioner object (1-parameter callable object) for CG, see L{SGSPreconditioner} for example
@var PRECONDITIONER: preconditioner object (1-parameter callable object) that returns
I{z} for a given I{r}
@var SGS_ITERATIONS: number of Symmetric Gauss-Seidel iterations
@var OPTIMIZE_SGS: whether to use Fortran code for Symmetric Gauss-Seidel
@var OPTIMIZE_AX: whether to use Fortran code for matrix-vector multiplication
@var STIFFNESS_FUNCTION: coefficient function that returns stiffness for any point (x,y)
"""

VISUALIZE = True
PRECONDITIONER_CLASS = None
PRECONDITIONER = None
SGS_ITERATIONS = None
OPTIMIZE_SGS = True
OPTIMIZE_AX = True
STIFFNESS_FUNCTION = None
