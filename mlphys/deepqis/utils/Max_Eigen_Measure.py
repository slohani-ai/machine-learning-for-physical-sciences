"""
author: Sanjaya Lohani
email: slohani@mlphys.com
Licence: Apache-2.0
"""
import tensorflow as tf


def max_eigen(dm_tensor):
    eigv, _ = tf.linalg.eigh(dm_tensor)
    eigv = tf.math.real(eigv)
    eigv_max = tf.reduce_max(eigv, axis=1)
    return eigv_max


def mean_eigen_order(dm_tensor):
    eigv, _ = tf.linalg.eigh(dm_tensor)
    eigv = tf.math.real(eigv)
    eigv_mean = tf.reduce_mean(tf.sort(eigv, axis=1), axis=0)
    return eigv_mean
