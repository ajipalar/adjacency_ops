import jax
import jax.numpy as np
from collections import namedtuple

JList = namedtuple("JList", "arr lead_idx")

def create_empty_jlist(length, dtype):
    """
    f :: (int -> dtype -> j)
    Creates an empty JList
    The array is implements with large default values
    """
    return JList(jnp.ones(length, dtype=dtype) * length + 2, lead_idx=0)

def append_jlist(jlist, val):
    """
    f :: (j -> v -> j)
    """
    arr = jlist.arr.at[jlist.lead_idx].set(val) 
    return JList(arr, jlist.lead_idx + 1)

def pop_jlist(jlist):
    """
    f :: (j -> (v, j))

    pop an element off the end of the list
    Returns:
      val : the popped element
      jlist : a jlist the same size as the original
    """
    val = jlist.arr[jlist.lead_idx-1]
    return val, JList(jlist.arr, jlist.lead_idx - 1)

def in_jlist(x, jlist) -> bool:
    """
    f :: (x -> j -> bool)
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
