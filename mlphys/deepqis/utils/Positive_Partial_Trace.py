"""
author: Sanjaya Lohani
email: slohani@mlphys_nightly.com
Licence: Apache-2.0
"""
'''
It only works for the two qubit case
'''
import numpy as np

__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"


def find(mat):
    aa, bb, cc, dd = mat[:2, :2], mat[:2, 2:], mat[2:, :2], mat[2:, 2:]
    e = np.eye(4)
    par_trans = np.kron(e[0].reshape(2, 2), aa.T) + np.kron(e[1].reshape(2, 2), bb.T) + np.kron(e[2].reshape(2, 2),
                                                                                                cc.T) + np.kron(
        e[3].reshape(2, 2), dd.T)
    return par_trans


def PPT(rho):  # dm should be rank 3 (n, dim, dim)
    '''
    return ent_rank, sep_rank; indices for the density matrices:
    ent_rank: index of entangled density matrix
    sep_rank: index of separable density matrix
    '''

    rho_tans_b = list(map(find, rho))
    rho_tans_b = np.array(rho_tans_b).reshape(-1, 4, 4)

    eigen = np.linalg.eigvals(rho_tans_b)  # [n, dim]\
    ent_rank = []
    sep_rank = []
    for i in range(len(eigen)):
        tol = 1e-15
        elements = np.where(np.abs(eigen[i]) < tol, 0, eigen[i])
        # print('eigen values', elements)
        if np.any(elements < 0):
            # print(f'rank {i} is found to be entangled.')
            ent_rank.append(i)
            # print(eigen[i])
        else:
            sep_rank.append(i)
            pass
    return ent_rank, sep_rank