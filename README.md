# AdjacencyOps

AdjacencyOps provides operations over simple adjacency matrices implemented in [JAX](https://github.com/google/jax). 

Such operations are designed to be fast, jitcompiled with [JAX].
They are not expected to scale well to _large_ matrices. 

Operations are pure functions with haskell-like call signatures.
(M-\>M)
In some cases shape or datatype information is required. Such as
`distance_matrix(A, Ashape)`  
For these cases shape arguments must be known at (jit) compile time.
This can be achieved through partial function application.
`f = jax.tree_util.Partial(distance_matrix, Ashape=shape)` 


Operations include:
- distance matrix
- $tr(A)$ 
- $\det(A)$ 
- $\deg(A)$ 
- The spectrumn of $A$
- _n_-walks
