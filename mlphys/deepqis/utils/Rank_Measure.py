"""
author: Sanjaya Lohani
email: slohani@mlphys.com
Licence: Apache-2.0
"""
import tensorflow as tf


def rank(dm_tesnor):
    ranks = tf.linalg.matrix_rank(dm_tesnor, tol=1e-5)
    return ranks.numpy()
