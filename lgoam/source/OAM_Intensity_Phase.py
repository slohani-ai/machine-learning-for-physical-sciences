'''
author: Sanjaya Lohani
email: slohani@mlphys.com
Started: 2017, updated to tensoflow 2.4 and published: 2020
'''

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import os


class LG_Lights_Tensorflow:

    def __init__(self,xpixel=600,ypixel=600,dT=8e-6,verbose=False):
        self.xpix = xpixel
        self.ypix = ypixel
        self.dT = dT
        x = tf.linspace(-self.xpix / 2, self.xpix / 2-1, self.xpix) * self.dT
        y = tf.linspace(-self.ypix / 2, self.ypix / 2-1, self.ypix) * self.dT
        self.xx, self.yy = tf.meshgrid(x, y)
        self.xx = tf.cast(self.xx, tf.complex128)
        self.yy = tf.cast(self.yy, tf.complex128)
        self.r = tf.math.sqrt(self.xx ** 2 + self.yy ** 2)
        self.verbose = verbose
        self.count_image = 0

    def Factorial(self,n):
        n = tf.cast(n,tf.float64)
        lgamma = tf.math.lgamma(n+1.)
        factorial = tf.cast(tf.exp(lgamma), tf.complex128)
        return factorial


    def LG_State(self,p_l_w):
        p, l, w = p_l_w[0], p_l_w[1], p_l_w[2]
        pi = tf.constant(tnp.pi, tf.complex128)
        # print (p,l,w)
        a_r = tf.math.sqrt(2./ pi) * tf.divide(1., w) * tf.exp(-tf.pow(self.r, 2) / tf.pow(w, 2))
        R = tf.math.sqrt(tf.constant(1., tf.complex128) / self.Factorial(tf.math.abs(l))) * \
            (self.r * tf.cast(tf.math.sqrt(2.),tf.complex128) / w) ** tf.cast(tf.math.abs(l), tf.complex128)
        phase_due_l = tf.math.exp(-tf.cast(1j,tf.complex128)*l*tf.cast(tf.math.angle(self.xx+1j*self.yy), tf.complex128))

        def p_0():
            if self.verbose:
                print('|Found P = 0, ... ...')
            return a_r*R*phase_due_l

        def p_1():
            if self.verbose:
                print('|Found P = 1, ... ...')
            return a_r*R*(-(2.+0j)*self.r**2/w**2 + tf.cast(tf.math.abs(l), tf.complex128) + (1.+0j))*phase_due_l
        psi = tf.cond(tf.equal(p, 0), p_0, p_1)

        return psi

    def Save_Image(self,intensity):
        self.count_image +=1
        inten_3d = tf.expand_dims(intensity, axis=-1)
        # print(inten_3d.shape)
        pil_image = tf.keras.preprocessing.image.array_to_img(inten_3d)
        pil_image.save(f'Image_{self.count_image}.png')
        return 1

    def Non_Superposition(self, p_l_array, w=0, grating_period=0, save_image=False, numpy_array=False):
        device = ['GPU' if tf.config.list_physical_devices('GPU') else 'CPU']
        if self.verbose:

            if device[0] == 'GPU':
                print('|Pysical devices found: ', tf.config.list_physical_devices('GPU'))
            else:
                print('|Running on CPU .. .. ..')

        with tf.device(device[0]):

            if isinstance(p_l_array, list):
                p_l_array = tf.reshape(p_l_array, [-1, 2])
                if self.verbose:
                    print('|list of p and l values are found .. ..')
            tf.debugging.assert_equal(p_l_array.shape[1], 2, \
                                      f'|p_l_array should be in [?,2] shape, but found in {p_l_array.shape}')

        w_mat = tf.reshape(tf.repeat(w, len(p_l_array)), [-1, 1])
        p_l_array = tf.cast(p_l_array, tf.float32)
        p_l_w_array = tf.concat([p_l_array, w_mat], axis=1)
        p_l_w_array = tf.cast(p_l_w_array, tf.complex128)

        psi_list = tf.map_fn(self.LG_State, p_l_w_array)

        psi_array = tf.reshape(psi_list, [-1, self.xpix, self.ypix])
        intensity_list = tf.math.square(tf.math.abs(psi_array))

        if grating_period == 0:
            phase_list = tf.math.angle(psi_array)
        else:
            phase = tf.reshape(tf.math.exp((2. * tnp.pi * self.xx / (grating_period*self.dT)) * 1j), [-1,self.xpix, self.ypix])\
                    * psi_array
            phase_list = tf.math.angle(phase)

        if save_image:
            path_to_image_dir = f'./OAM_Non_Sup_Images'

            if not os.path.exists(path_to_image_dir):
                os.mkdir(path_to_image_dir)
            cur = os.getcwd()
            goto = os.path.join(cur, path_to_image_dir)
            os.chdir(goto)
            total_count = tf.map_fn(self.Save_Image, intensity_list)
            total_count = tf.reduce_sum(total_count)

            if self.verbose:
                print (f'|Total {total_count} images are saved.')
            os.chdir(cur)
        intensity_list = tf.cast(intensity_list, tf.float32)
        phase_list = tf.cast(phase_list, tf.float32)

        if numpy_array:
            intensity_list = intensity_list.numpy()
            phase_list = phase_list.numpy()

        return intensity_list, phase_list


    def Superposition(self, p_l_array, alpha_array, w=0, grating_period=0, save_image=False, rank=0,numpy_array=False):
        device = ['GPU' if tf.config.list_physical_devices('GPU') else 'CPU']

        if self.verbose:
            if device[0] == 'GPU':
                print ('|Physical devices found: ', tf.config.list_physical_devices('GPU'))
            else:
                print ('|Running on CPU .. .. ..')

        with tf.device(device[0]):

            if isinstance(p_l_array, list):
                p_l_array = tf.reshape(p_l_array, [-1, 2])
                if self.verbose:
                    print ('|list of p and l values are found .. ..')

            if isinstance(alpha_array, list):
                alpha_array = tf.reshape(alpha_array, [-1, 1])
                if self.verbose:
                    print ('|list of alpha is found .. ..')

            tf.debugging.assert_equal(tf.shape(p_l_array)[1], 2, \
                            f'|p_l_array should be in [?,2] shape, but found in {p_l_array.shape}')

            w_mat = tf.reshape(tf.repeat(w, len(p_l_array)), [-1, 1])
            p_l_array = tf.cast(p_l_array, tf.float32)
            p_l_w_array = tf.concat([p_l_array, w_mat], axis=1)
            p_l_w_array = tf.cast(p_l_w_array, tf.complex128)
            psi_list = tf.map_fn(self.LG_State, p_l_w_array)#,dtype=tf.complex128)

            psi_array = tf.reshape(psi_list, [-1,self.xpix, self.ypix])
            alpha_array = tf.cast(tf.reshape(alpha_array, [-1, 1, 1]), tf.complex128)
            sup = tf.math.reduce_sum(psi_array*alpha_array, axis=0)
            intensity = tf.math.square(tf.math.abs(sup)) #(400,400)

            if grating_period == 0:
                phase = tf.math.angle(sup)
                # phase = tf.where(phase < 0, -phase, phase)
                # phase = tf.where(phase < 1e-14, 0, phase)

            else:
                phase = tf.math.exp((2. * tnp.pi * self.xx / (grating_period*self.dT)) * 1j) * sup
                phase = tf.math.angle(phase)
                # phase = tf.where(phase < 0, -phase, phase)
                # phase = tf.where(phase < 1e-14, 0, phase)

            if save_image:
                path_to_image_dir = f'./OAM_Sup_Images_Rank_{rank}'

                if not os.path.exists(path_to_image_dir ):
                    os.mkdir(path_to_image_dir)
                cur = os.getcwd()
                goto = os.path.join(cur, path_to_image_dir )
                os.chdir(goto)

                if tf.rank(intensity) !=3:
                    intensity = tf.expand_dims(intensity, axis=0)
                total_count = tf.map_fn(self.Save_Image, intensity)
                total_count = tf.reduce_sum(total_count)

                if self.verbose:
                    print(f'|Total {total_count} images are saved.')
                os.chdir(cur)
            intensity = tf.cast(intensity, tf.float32)
            if numpy_array:
                intensity = intensity.numpy()
                phase = phase.numpy()

            return intensity, phase

    def Superposition_Batch(self,p_l_array, alpha_array, w=0, grating_period=0, save_image=False,numpy_array=False):
        tf.debugging.assert_equal(tf.rank(p_l_array), 3, 'p_l_array should be 3 dim')
        tf.debugging.assert_equal(tf.rank(alpha_array), 3, 'alpha_array should be 3 dim')
        intensity_list = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        phase_list = tf.TensorArray(dtype=tf.float64, size=1, dynamic_size=True)

        for i in range(len(p_l_array)):
            string_value = 'Modes_'

            for k in range(len(p_l_array[i])):
                string_value +=str(p_l_array[i][k])
            rank_value = string_value #str(p_l_array[i][0]) + str(p_l_array[i][1])

            inten, phase = self.Superposition(p_l_array=p_l_array[i], alpha_array=alpha_array[i],\
                                             w=w, grating_period=grating_period,\
                                             save_image=save_image,\
                                             rank=rank_value,numpy_array=numpy_array)
            intensity_list.write(i,inten).mark_used()
            phase_list.write(i,phase).mark_used()

        intensity_list = tf.squeeze(intensity_list.stack())
        phase_list = tf.squeeze(phase_list.stack())
        if numpy_array:
            intensity_list = intensity_list.numpy()
            phase_list = phase_list.numpy()

        return intensity_list, phase_list
