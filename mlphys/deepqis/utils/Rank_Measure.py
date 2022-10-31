"""
author: Sanjaya Lohani
email: slohani@mlphys_nightly.com
Licence: Apache-2.0
"""
import tensorflow as tf

__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"

def rank(dm_tesnor):
    ranks = tf.linalg.matrix_rank(dm_tesnor, tol=1e-5)
    return ranks.numpy()
