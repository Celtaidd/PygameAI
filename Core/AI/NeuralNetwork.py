import math
import numpy as np
import tensorflow as tf
import numba as nb

@nb.jit(forceobj=True)
def ELu(in_value):
    in_value[in_value <= 0] = math.e ** in_value[in_value <= 0] - 1
    return in_value

@nb.jit(forceobj=True)
def RELu(in_array):
    mask = in_array <= 0
    inds = tf.where(mask)
    updates = tf.boolean_mask(in_array, mask)
    updates = updates * 0.1
    res = tf.tensor_scatter_nd_update(in_array, inds, updates)
    return res

@nb.jit(forceobj=True)
def TELu(in_array):
    mask = in_array <= 0
    inds = tf.where(mask)
    updates = tf.boolean_mask(in_array, mask)
    updates = math.e ** updates - 1.0
    res = tf.tensor_scatter_nd_update(in_array, inds, updates)
    return res


def BELu(in_value):
    in_value[in_value > 0] = 1
    in_value[in_value <= 0] = math.e ** in_value[in_value <= 0]
    return in_value

@nb.jit(forceobj=True)
def linear_propagation(A_prev, W, b):
    return np.array(np.dot(W, A_prev) + b).sum(axis=0)

@nb.jit(forceobj=True)
def Tlinear_propagation(A_prev, W, b):
    return tf.reduce_sum(tf.tensordot(W, A_prev, axes=1) + b, axis=0)

def initialize_parameters(layers_dims, warmup=False):
    if not warmup:
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 2.0 - 1
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1,))
    else:
        parameters = warmup
    
    return parameters

# Endfile
