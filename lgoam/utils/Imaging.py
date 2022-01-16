'''
author: Sanjaya Lohani
email: slohani@mlphys.com
'''

import tensorflow as tf
import os
from datetime import datetime

class Save:

    def __init__(self):
        self.count = 0

    def Array_To_Image(self, tensor):
        #tensor should be [w,h,channel]
        tensor = tf.expand_dims(tensor, axis=-1)
        img = tf.keras.preprocessing.image.array_to_img(tensor)
        img.save(f'Image_{self.count}.png')
        self.count+=1
        return 0

    def Create_Dir(self, folder=None):
        cur = os.getcwd()
        now = datetime.now().strftime("%H:%M:%S").split(':')
        now = ('_').join(now)

        if folder == None:
            goto = os.path.join(cur, f'default_dir_{now}')
        else:
            goto = os.path.join(cur, folder)

        if not os.path.exists(goto):
            os.mkdir(goto)

        return  goto, cur


    def Save_Tensor_Image(self, tensor_array, folder=None):

        if tf.rank(tensor_array) <=3:
            tf.print('|Tensor is either rank 3 or 2, so only one folder is created.')

            if tf.rank(tensor_array) == 2:
                tensor_array = tf.expand_dims(tensor_array, axis=0)
            goto,cur = self.Create_Dir(folder=folder)
            os.chdir(goto)
            tf.map_fn(self.Array_To_Image,tensor_array)
            os.chdir(cur)

        elif tf.rank(tensor_array) == 4:
            tf.print ('|Rank 4 tensor is found. {} folders are created.'.format(tf.shape(tensor_array)[0]))

            for j in tensor_array:
                goto, cur = self.Create_Dir(folder=folder)
                os.chdir(goto)
                tf.map_fn(self.Array_To_Image, j)
                os.chdir(cur)






