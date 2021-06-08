import numpy as np
import tensorflow as tf


def nextpow2(x):
    p2 = np.ceil(np.log2(x))
    return int(np.power(2.0, p2))


def complex_abs(x):
    return tf.math.conj(x) * x
