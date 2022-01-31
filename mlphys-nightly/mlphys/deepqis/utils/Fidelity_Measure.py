import tensorflow as tf


def Fidelity_Metric(rhos_pred_value, rhos_true_value):
    rho_mat_pred = rhos_pred_value
    sqrt_rho_pred = tf.linalg.sqrtm(rho_mat_pred)
    products = tf.matmul(tf.matmul(sqrt_rho_pred, rhos_true_value), sqrt_rho_pred)
    fidelity = tf.square(tf.linalg.trace(tf.math.real(tf.linalg.sqrtm(products))))
    av_fid = tf.reduce_mean(fidelity)
    return fidelity, av_fid
