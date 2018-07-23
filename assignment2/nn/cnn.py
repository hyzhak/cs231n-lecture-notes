import numpy as np
import tensorflow as tf


def cnn(idx, X, filters, kernel_size, is_training,
        strides=(1, 1, 1, 1), padding='SAME', use_batchnorm=True,
        dropout=None):
    input_filter_size = X.shape[-1]
    Wconv = tf.get_variable(f'Wconv{idx}', shape=[*kernel_size, input_filter_size, filters])
    bconv = tf.get_variable(f'bconv{idx}', shape=[filters])
    out = tf.nn.conv2d(X, filter=Wconv, strides=strides, padding=padding) + bconv

    # ReLU Activation Layer
    out = tf.nn.relu(out)

    # Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
    # axis=3 channel axis
    if use_batchnorm:
        out = tf.layers.batch_normalization(out, axis=3, training=is_training)

    if dropout is not None:
        out = tf.layers.dropout(out, rate=dropout, training=is_training)

    return out, [Wconv, bconv]
