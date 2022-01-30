
"""
Created on Fri Feb 21 13:27:32 2020

@author: Sanjaya_lohani (slohani@mlphys.com)
"""

import tensorflow as tf
import numpy as np



class PredictDensityMatrix(tf.keras.layers.Layer):
    def __init__(self, name='Density_matrix', qubit_size=2, **kwargs):
        super(PredictDensityMatrix, self).__init__(name=name, **kwargs)

        self.qubit_size = qubit_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'qubit_size': self.qubit_size
        })
        return config

    def diag_and_off_diag(self, t_off_values):  # [1,2]
        t_off_values = tf.cast(t_off_values, tf.float32)
        concat = tf.cast(tf.complex(t_off_values[0], t_off_values[1]), tf.complex128)
        return concat

    def t_to_mat_layer(self, t):
        indices = list(zip(*np.tril_indices(n=2 ** self.qubit_size, k=-1)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)
        diags = tf.cast(t[:2 ** self.qubit_size], tf.float32)
        off_diags = tf.reshape(t[2 ** self.qubit_size:], [-1, 2])
        real, imag = tf.split(tf.cast(off_diags, tf.float32), 2, 1)
        real = tf.reshape(real, [-1])
        imag = tf.reshape(imag, [-1])

        t_mat_real = tf.sparse.SparseTensor(indices=indices, values=real,
                                            dense_shape=[2 ** self.qubit_size, 2 ** self.qubit_size])
        t_mat_imag = tf.sparse.SparseTensor(indices=indices, values=imag,
                                            dense_shape=[2 ** self.qubit_size, 2 ** self.qubit_size])

        t_mat_list_real = tf.sparse.to_dense(t_mat_real)
        t_mat_array_real_with_diag = tf.cast(tf.linalg.set_diag(t_mat_list_real, diags), tf.complex128)
        t_mat_array_imag_with_no_diag = tf.cast(tf.sparse.to_dense(t_mat_imag), tf.complex128)
        t_mat_array = tf.add(t_mat_array_real_with_diag, 1j * t_mat_array_imag_with_no_diag)

        return t_mat_array

    def t_mat_rho_layer(self, t_mat):
        rho = tf.matmul(t_mat, tf.transpose(t_mat, conjugate=True)) / \
              tf.linalg.trace(tf.matmul(t_mat, tf.transpose(t_mat, conjugate=True)))
        return rho

    def call(self, inputs):
        logits_l = tf.cast(inputs, tf.complex128)
        # t_matrix_list_l = tf.map_fn(self.t_to_mat_layer,logits_l/100)
        t_matrix_list_l = tf.vectorized_map(self.t_to_mat_layer, logits_l / 100)
        t_matrix_l = tf.reshape(t_matrix_list_l, [-1, 2 ** self.qubit_size, 2 ** self.qubit_size])
        # rho_mat_pred = tf.map_fn(self.t_mat_rho_layer,t_matrix_l)
        rho_mat_pred = tf.vectorized_map(self.t_mat_rho_layer, t_matrix_l)
        return rho_mat_pred


class ErrorNode(tf.keras.layers.Layer):
    def __init__(self, name='Error', **kwargs):
        super(ErrorNode, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return inputs
