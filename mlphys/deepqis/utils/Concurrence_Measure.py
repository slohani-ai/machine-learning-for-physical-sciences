"""
author: Sanjaya Lohani
email: slohani@mlphys.com
Licence: Apache-2.0
"""
import numpy as np
import scipy

__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"

def concurrence_single(dm):
    yy_mat = np.fliplr(np.diag([-1, 1, 1, -1]))
    sigma = dm.dot(yy_mat).dot(dm.conj()).dot(yy_mat)
    w = np.sort(np.real(scipy.linalg.eigvals(sigma)))
    w = np.sqrt(np.maximum(w, 0.0))
    con = max(0.0, w[-1] - np.sum(w[0:-1]))
    return con


def concurrence(dm_tensor):
    con_list = list(map(concurrence_single, dm_tensor))
    con_tensor = np.array(con_list)
    return con_tensor
