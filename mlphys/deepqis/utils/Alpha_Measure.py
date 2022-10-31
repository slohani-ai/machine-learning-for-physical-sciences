"""
author: Sanjaya Lohani
email: slohani@mlphys_nightly.com
Licence: Apache-2.0
"""

__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"

def alpha(D, K, mean):
    alpha_x = D * (mean - 1) / (D + K - 1 - mean * K * D)
    return alpha_x
