import numpy as np

import mlphys.deepqis.utils.gen_basis_order as gen_basis_order


class MultiQubitSystem:

    def __init__(self, qubit_size=4):
        self._qs = qubit_size
        self.H = np.array([1., 0.]).astype(np.float32).reshape(2, 1)
        self.V = np.array([0., 1.]).astype(np.float32).reshape(2, 1)
        self.D = 1 / np.sqrt(2.) * (self.H + self.V)
        self.A = 1 / np.sqrt(2.) * (self.H - self.V)
        self.R = 1 / np.sqrt(2.) * (self.H + 1j * self.V)
        self.L = 1 / np.sqrt(2.) * (self.H - 1j * self.V)
        self.h = np.matmul(self.H, np.conjugate(self.H.T))
        self.v = np.matmul(self.V, np.conjugate(self.V.T))
        self.d = np.matmul(self.D, np.conjugate(self.D.T))
        self.a = np.matmul(self.A, np.conjugate(self.A.T))
        self.r = np.matmul(self.R, np.conjugate(self.R.T))
        self.l = np.matmul(self.L, np.conjugate(self.L.T))
        self.dict_proj = {'h': self.h, 'v': self.v, 'd': self.d, 'a': self.a, 'r': self.r, 'l': self.l}

    def Kron_Povm(self, povm):
        prod = 0.
        if len(povm) == 1:
            prod = self.dict_proj[povm[0]]
        else:
            prod = np.kron(self.dict_proj[povm[0]], self.dict_proj[povm[1]])
            for j in range(2, self._qs):
                prod = np.kron(prod, self.dict_proj[povm[j]])
        return prod

    def NISQ_Projectors(self):
        ibmq_proj = gen_basis_order.Basis_Order(qs=self._qs)
        # print(ibmq_proj)
        Proj = list(map(self.Kron_Povm, ibmq_proj))
        Proj = np.array(Proj).reshape(-1, 2 ** self._qs, 2 ** self._qs)
        return Proj
