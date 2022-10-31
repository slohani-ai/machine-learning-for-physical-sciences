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

def purity(dm):
    mul = tf.math.real(tf.linalg.trace(tf.linalg.matmul(dm, dm, adjoint_b=True)))
    return mul.numpy()
