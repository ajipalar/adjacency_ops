import jax
import jax.numpy as jnp
from collections import namedtuple

init = namedtuple("init", "arr tail_idx length")

def zeros(length, dtype):
    # Queue
    # The queue has a size equal to the number of elements in the queue
    # The queue has a length equal to len(arr)
    arr = jnp.zeros(length, dtype=dtype)
    # The empty queue's leader is at position 0 
    leader_idx = 0
    # The empty queue's tail is at position 0

    return init(arr=arr, tail_idx=0, length=length)

def enqueue(q, val):
    """
    A queue is a first in first out data structure
    This function is meant to traced by JAX and jit compiled 
    If the queue is full (q.length) and enqueue is called,
    enqeue_jax is undefined and the queue is invalid
    """
    return init(q.arr.at[q.tail_idx].set(val), q.tail_idx + 1, q.length)

def dequeue(q):
    """
    dequeue a Queue. traceable 
    """
    arr = jax.lax.fori_loop(1, q.length, lambda i, arr: arr.at[i-1].set(arr[i]), q.arr) 
    return q.arr[0], init(arr, q.tail_idx-1, q.length)



