import numpy as np
import tensorflow as tf


def affine(idx, X, output_size):
    input_size = np.prod(X.shape[1:])
    W = tf.get_variable(f'W{idx}', shape=[input_size, output_size])
    b = tf.get_variable(f'b{idx}', shape=[output_size])
    if len(X.shape) > 3:
        out = tf.reshape(X, [-1, input_size])
    else:
        out = X
    out = tf.matmul(out, W) + b
    return out, [W, b]
