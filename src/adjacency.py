import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

shape = (4, 4)
A = np.zeros(shape)
A[0, 1] = 1
A[0, 2] = 1
A = A + A.T
A = jnp.array(A)
S = A.copy()

V = np.zeros(shape)
V[0, 1] = 1
V[1, 2] = 1
V[2, 3] = 1
V = V + V.T

def eigh(A):
    a, v = jnp.linalg.eigh(A)
    a = a.real
    v = v.real
    a = jnp.sort(a)
    return a

def _body_npow(i, val):
    An, A = val
    An = An @ A
    return An, A

def npow(A, n):
    """
    Raise the matrix A to the nth power.
    For an adjacency matrix this corresponds to the number
    of paths traversible between vertex i and vertex j.
    """
    An, A = jax.lax.fori_loop(0, n-1, _body_npow, (A, A)) 
    return An

def tr(A):
    """
    The trace of the adjacency matrix must be all 0
    """
    return jnp.trace(A)

def det(A):
    """
    The determinant of the adjacency matrix is equal to
    """
    return jnp.linalg.det(A)

def _body_distance_matrix(n, val):
    A, S = val
    An = npow(A, n) 
    S = jnp.where(S==0,(An>0)*(n), S)
    return A, S

def distance_matrix(A, Ashape):
    """
    Returns a matrix S whose off diagonal elements i,j
    represent the shortest path length between elements i,j.
    Where 0 is no path, 1 is an edge, 2 is 2 away etc.

    Returns
    - S. The geodesic distance matrix of A where 0 represents
         infinite distance.
    - A. The n-1th matrix power of the adjacency matrix. 
    
    """
    nrows, ncols = Ashape
    S = A.copy() 
    A, S = jax.lax.fori_loop(2, nrows, _body_distance_matrix, (A, S))
    return S, A
    


g = jax.jit(Partial(distance_matrix, Ashape=shape))


    




