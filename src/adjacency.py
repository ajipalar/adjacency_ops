import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from collections import namedtuple
from typing import NamedTuple, Any, Callable

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

AdjacentNodes = namedtuple("AdjacentNodes", "arr n_adjacent_nodes")
Queue = namedtuple("Queue", "arr tail_idx length")
JList = namedtuple("JList", "arr lead_idx")

def create_empty_jlist(length, dtype):
    """
    Creates an empty JList
    The array is implements with large default values
    """
    return JList(jnp.ones(length, dtype=dtype) * length + 2, lead_idx=0)

def append_jlist(jlist, val):
    arr = jlist.arr.at[jlist.lead_idx].set(val) 
    return JList(arr, jlist.lead_idx + 1)

def pop_jlist(jlist):
    """
    pop an element off the end of the list
    Returns:
      val : the popped element
      jlist : a jlist the same size as the original
    """
    val = jlist.arr[jlist.lead_idx-1]
    return val, JList(jlist.arr, jlist.lead_idx - 1)

def in_jlist(x, jlist) -> bool:
    """
    Tests if the value x is in jlist
    """

    lead_idx = jlist.lead_idx
    in_ = False
    def body(i, val):
        in_ , jlist , x= val
        in_ = (x == jlist.arr[i]) | in_
        return in_, jlist, x

    val = in_, jlist, x
    val = jax.lax.fori_loop(0, lead_idx, body, val) 
    in_, jlist, x = val
    return in_


def eigh(A):
    """
    eigh :: (a -> v)
    Returns the sorted eigenvalues of the adjacency matrix A
    """
    a, v = jnp.linalg.eigh(A)
    a = a.real
    v = v.real
    a = jnp.sort(a)
    return a

def _body_npow(i, val):
    An, A = val
    An = An @ A
    return An, A

def npow(A, n: int):
    """
    npow :: (a -> int -> a)
    Raise the matrix A to the nth power.
    For an adjacency matrix this corresponds to the number
    of paths traversible between vertex i and vertex j.
    """
    An, A = jax.lax.fori_loop(0, n-1, _body_npow, (A, A)) 
    return An

def tr(A) -> float:
    """
    tr :: (a -> float)
    The trace of the adjacency matrix must be all 0
    """
    return jnp.trace(A)

def det(A) -> float:
    """
    det :: (a -> float)
    The determinant of the adjacency matrix is equal to
    """
    return jnp.linalg.det(A)

def degv(A):
    """
    degv :: (a -> v)
    """
    return A.sum(axis=0)

def deg(A, i):
    return degv(A)[i]

def _body_distance_matrix(n, val):
    A, S = val
    An = npow(A, n) 
    S = jnp.where(S==0,(An>0)*(n), S)
    return A, S

def distance_matrix(A, Ashape):
    """
    distance_matrix :: (a -> (int, int) -> (a, a))

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




def X2Y(X, Y, Y_boolean_array_fun, true_fun, false_fun, init): 
    """
    X2Y :: (x -> y -> (x->y->a->x[bool]) -> (x->y->a->y) -> (x->y->a->y) -> y) 
    General matrix update. Updates the values of
    matrix X based on the values of matrix Y.

    The variable init is used to carry any additional information

    If x is an array then x[bool] is an array of the same shape with
    datatype bool.

    X :: x - the reference matrix
    Y :: y - the update matrix
    init :: a - any additional information
    Y_boolean_array_fun :: (x->y->a-> x[bool])
    true_fun :: (x->y->a->y)
    false_fun :: (x->y->a->y)
    """

    Y = jnp.where(Y_boolean_array_fun(X, Y, init), true_fun(X, Y, init), false_fun(X, Y, init))
    return Y 

def G2L(G, L, L_boolean_array_fun, true_fun, false_fun, init):
    """
    General matrix update. Updates the values of
    matrix G based on the values of matrix L.

    The variable init is used to carry any additional information

    G :: M[x, x, dx] - the reference matrix
    L :: Y[y, y, dy] - the update matrix
    G_boolean_array_fun :: (M[x, x]->M[y, y]->M[y, y, dbool]) 
    true_fun :: (M[x, x] -> M[y, y] -> M[y, y])
    false_fun :: (M[x, x] -> M[y, y] -> M[y, y]
    init :: Any
    """
    L = jnp.where(L_boolean_array_fun(G, L, init), true_fun(G, L, init), false_fun(G, L, init))
    return L

def L2G(L, G, G_boolean_array_fun, true_fun, false_fun, init):
    """

    """
    G = jnp.where(G_boolean_array_fun(L, G, init), true_fun(L, G, init), false_fun(L, G, init))
    return G

class FInvPair(NamedTuple):
    """
    A function and its inverse

    fwd :: (a -> b)
    inv :: (b -> a)
    """
    fwd: Callable 
    inv: Callable 


def bfs_ref(A, root: int):
    """
    A breadth first search

    Params:
      A an adjacency matrix
    Returns:
      explored: the set of nodes connected to the root
    """
    _assert_A(A)
    assert type(root) in [np.int8, np.int32, np.int64, int] 

    n = len(A)

    explored = set()
    explored.add(root)

    m = n * (n - 1) // 2

    # a queue of nodes
    q = deque([root], maxlen=n)

    while len(q) > 0:
        v = q.popleft()
        adjacent_nodes = get_adjacent_nodes(A, v)
        for node in adjacent_nodes:
            if node not in explored:
                explored.add(node)
                q.append(node)
    return explored

def create_empty_queue(length, dtype):
    # Queue
    # The queue has a size equal to the number of elements in the queue
    # The queue has a length equal to len(arr)
    arr = jnp.zeros(length, dtype=dtype) # The empty queue's leader is at position 0 
    leader_idx = 0
    # The empty queue's tail is at position 0

    return Queue(arr=arr, tail_idx=0, length=length)

def enqueue_jax(q, val):
    """
    A queue is a first in first out data structure
    This function is meant to traced by JAX and jit compiled 
    If the queue is full (q.length) and enqueue_jax is called,
    enqeue_jax is undefined and the queue is invalid
    """
    return Queue(q.arr.at[q.tail_idx].set(val), q.tail_idx + 1, q.length)

def dequeue_jax(q):
    """
    dequeue a Queue. traceable 
    """
    arr = jax.lax.fori_loop(1, q.length, lambda i, arr: arr.at[i-1].set(arr[i]), q.arr) 
    return q.arr[0], Queue(arr, q.tail_idx-1, q.length)




BFS_WHILE_PARAMS = namedtuple("BFS_WHILE_PARAMS",
                              "explored q A len_A")

def _bfs_jax_while_loop_piece(explored, q, A, len_A):
    body_fun = Partial(_bfs_jax_while_body, len_A=len_A)
    return jax.lax.while_loop(
            lambda x: x[1].tail_idx > 0,
            lambda x: body_fun(*x),
            (explored, q, A))


def _bfs_jax_while_body(explored, q, A, len_A):
#    jax.debug.breakpoint()
    v, q = dequeue_jax(q)

    adj = get_adjacent_nodes_jax(A, v, len_A)
    _, explored, q, adj = _bfs_jax_fori_loop_piece(
            adj, explored, q)
#    jax.debug.breakpoint()

    return explored, q, A 



def _bfs_jax_fori_loop_body(i, val):
    """ (node, (explored, q)) -> (explored, q) """
    _, explored, q, adj = val
    node = adj.arr[i]
    init = node, explored, q
    #jax.debug.breakpoint()

    in_: bool = in_jlist(node, explored) 
    init = jax.lax.cond(
        in_, 
        lambda x: init, # do nothing
        lambda x: _bfs_jax_true_fun(*init), # update explored  q
        init)
    _, explored, q = init
    #jax.debug.breakpoint()
    return node, explored, q, adj

def _bfs_jax_fori_loop_piece(adj, explored, q):
    """ (adj, explored, q) -> (explored, q)"""
    val = 0, explored, q, adj
    return jax.lax.fori_loop(0, adj.n_adjacent_nodes,
                             _bfs_jax_fori_loop_body,
                             val)

def _bfs_jax_true_fun(node: int, explored, q):
    """ (T) -> (T) """
    explored = append_jlist(explored, node)
    q = enqueue_jax(q, node)
    return node, explored, q

def get_adjacent_nodes_jax(A, node_index: int, len_A, node_dtype=jnp.int32):
    """
    Traceable version of `get_adjacent_nodes`
    
    Params:
      A - an adjacency matrix
      node_index - the node from which adjacent nodes are to be found
      len_A - the number of rows/columns in A
    Returns:
      AdjacentNodes: namedtuple  
        arr: a sorted_array of adjacent_nodes
        n_adjacent_nodes: the number of adjacent nodes
    """

    # The max size is len_A-1
    # The garbage values are placed at the end of the array
    # The garbage values are  len_A * 9999 + 1

    default_values = len_A + 1 

    adjacent_nodes = jnp.ones(len_A-1, dtype=node_dtype) * default_values

    left_to_diag_len = node_index
    diag_to_bot_len = len_A - node_index - 1
    # Get the adjacent nodes by row
    
    def by_row(i, val):
        adjacent_nodes, A = val
        query_val = A[node_index, i]
        adj_val = jax.lax.cond(query_val == 1,
                               lambda i: i,
                               lambda i: default_values,
                               i)

        adjacent_nodes = adjacent_nodes.at[i].set(adj_val)
        return adjacent_nodes, A

    adjacent_nodes, A = jax.lax.fori_loop(0, left_to_diag_len, by_row, (adjacent_nodes, A))
    adjacent_nodes = jnp.sort(adjacent_nodes)

    n_adjacent_nodes = jnp.sum(adjacent_nodes != default_values) 
    return AdjacentNodes(adjacent_nodes, n_adjacent_nodes)


    
def bfs_jax(A, root: int, len_A, node_dtype=jnp.int32):
    """
    A jittable breadth first search

    Params:
      A: an adjacency matrix
    Returns:
      explored: the set of nodes connected to the root
    """
    
    # An array of explored nodes
    explored = create_empty_jlist(len_A, dtype=node_dtype) 

    # Assign the bait (root) as found
    explored = append_jlist(explored, root)


    # While the queue is not empty
#    jax.lax.while_loop(lambda q: q.tail_index != 0, while_loop_body, 

    q = create_empty_queue(length=len_A, dtype=node_dtype)
    q = enqueue_jax(q, root)

    explored, q, A = _bfs_jax_while_loop_piece(explored, q, A, len_A)
#    jax.debug.breakpoint()
    return explored
