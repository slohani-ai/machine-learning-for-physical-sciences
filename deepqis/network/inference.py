import tensorflow as tf
import pkg_resources
from deepqis.utils import Extract_Net as arlnet
import os

def load():
    """ Return pre-trained model"""
    path_to_model_, file_name = os.path.split(__file__)
    path_to_model = os.path.join(path_to_model_, 'models/ARL_ONRL_meas_n_1000_qubits_2_batch_4_alpha_0.4_BEST.h5')
    # stream = pkg_resources.resource_stream(path_to_model)
    return path_to_model

def fit(input_data):

    loaded_file = load()
    model = tf.keras.models.load_model(loaded_file, custom_objects={'ErrorNode':arlnet.ErrorNode, \
                                        'PredictDensityMatrix':arlnet.PredictDensityMatrix})

    def scaling_mean_0_std_1(row_matrix):
        m = tf.math.reduce_mean(row_matrix)
        std = tf.math.reduce_std(row_matrix)
        scaled = (row_matrix - m) / std
        return scaled
    x_test = tf.vectorized_map(scaling_mean_0_std_1, input_data)
    x_in = tf.reshape(x_test, [-1, 6, 6, 1])
    logits, dm_pred = model(x_in)

    return dm_pred, model
