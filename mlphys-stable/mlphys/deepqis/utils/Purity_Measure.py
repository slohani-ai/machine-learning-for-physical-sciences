import tensorflow as tf


def purity(dm):
    mul = tf.math.real(tf.linalg.trace(tf.linalg.matmul(dm, dm, adjoint_b=True)))
    return mul.numpy()
