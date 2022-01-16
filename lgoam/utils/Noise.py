'''
author: Sanjaya Lohani
email: slohani@mlphys.com
'''

import tensorflow as tf
import tensorflow_probability as tfp


class Noise_Dist:

    def __init__(self):
        self._distribution = tfp.distributions

    '''
    tensor_array should be a rank 2 tensor eg. [xpix,ypix].
    '''
    def Gaussian_Noise(self, tensor_array, mean=0, std=1, multiple=1, factor=1e5):
        tf.debugging.assert_rank(tensor_array, 2, '|Rank error, input tensor should be rank 2')
        factor = tf.constant(factor, tf.float32)
        xpix = tf.shape(tensor_array)[0]
        ypix = tf.shape(tensor_array)[1]
        gaussian_dist = self._distribution.Normal(loc=mean, scale=std)
        tensor_array_tiled = tf.reshape(tf.tile(tensor_array, [multiple, 1]), [-1, xpix, ypix])
        noisy_intensity = tf.add(tensor_array, factor*gaussian_dist.sample(tf.shape(tensor_array_tiled)))
        #noisey intensity will be rank 3 tensor [multiple,xpix,ypix]
        return noisy_intensity

    def Poisson_Noise(self, tensor_array, rate, multiple=1, factor=1.):
        tf.debugging.assert_rank(tensor_array, 2, '|Rank error, input tensor should be rank 2')
        factor = tf.constant(factor, tf.float32)
        xpix = tf.shape(tensor_array)[0]
        ypix = tf.shape(tensor_array)[1]
        poisson_dist = self._distribution.Poisson(rate=rate)
        tensor_array_tiled = tf.reshape(tf.tile(tensor_array, [multiple, 1]), [-1, xpix, ypix])
        noisy_intensity = tf.add(tensor_array, factor * poisson_dist.sample(tf.shape(tensor_array_tiled)))
        return noisy_intensity

    def Gamma_Noise(self, tensor_array, alpha, beta=None, multiple =1, factor=1.):
        tf.debugging.assert_rank(tensor_array, 2, '|Rank error, input tensor should be rank 2')
        factor = tf.constant(factor, tf.float32)
        xpix = tf.shape(tensor_array)[0]
        ypix = tf.shape(tensor_array)[1]
        gamma_dist = self._distribution.Gamma(concentration=alpha, rate=beta)
        tensor_array_tiled = tf.reshape(tf.tile(tensor_array, [multiple, 1]), [-1, xpix, ypix])
        noisy_intensity = tf.add(tensor_array, factor * gamma_dist.sample(tf.shape(tensor_array_tiled)))
        return noisy_intensity

    '''
    tensor_array_batch should be a rank 3 tensor eg. [batch,xpix,ypix].
    '''
    def Guassian_Noise_Batch(self, tensor_array_batch, mean=0, std=1, multiple=1, factor=1e5):
        tf.debugging.assert_rank(tensor_array_batch, 3, '|Rank error, input tensor should be rank 3')
        intensity_list = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        for i in range(len(tensor_array_batch)):
            inten = self.Gaussian_Noise(tensor_array=tensor_array_batch[i], mean=mean,\
                                     std=std, multiple=multiple, factor=factor)
            intensity_list.write(i, inten).mark_used()

        intensity_array = tf.squeeze(intensity_list.stack())
        return intensity_array

    def Poisson_Noise_Batch(self, tensor_array_batch, rate, multiple=1, factor=1.):
        tf.debugging.assert_rank(tensor_array_batch, 3, '|Rank error, input tensor should be rank 3')
        intensity_list = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        for i in range(len(tensor_array_batch)):
            inten = self.Poisson_Noise(tensor_array=tensor_array_batch[i], rate=rate, multiple=multiple, factor=factor)
            intensity_list.write(i, inten).mark_used()

        intensity_array = tf.squeeze(intensity_list.stack())
        return intensity_array

    def Gamma_Noise_Batch(self, tensor_array_batch, alpha, beta=None, multiple=1, factor=1.):
        tf.debugging.assert_rank(tensor_array_batch, 3, '|Rank error, input tensor should be rank 3')
        intensity_list = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        for i in range(len(tensor_array_batch)):
            inten = self.Gamma_Noise(tensor_array=tensor_array_batch[i], alpha=alpha, beta=beta,\
                                     multiple=multiple, factor=factor)
            intensity_list.write(i,inten).mark_used()
            
        intensity_array = tf.squeeze(intensity_list.stack())
        return intensity_array





