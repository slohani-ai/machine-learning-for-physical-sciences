import tensorflow as tf

def rank(dm_tesnor):
    ranks = tf.linalg.matrix_rank(dm_tesnor,tol=1e-5)
    return ranks