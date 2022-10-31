"""
author: Sanjaya Lohani
email: slohani@mlphys.com
Licence: Apache-2.0
"""

import numpy as np
from numpy.random import default_rng
from scipy import stats



__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"


class Randomness:

    def __init__(self, qs, seed=None):
        self._qs = qs
        self.seed = seed

        self.random_state = default_rng()
        if self.seed is not None:
            if isinstance(self.seed, np.random.Generator):
                self.random_state = seed
            else:
                self.random_state = default_rng(seed)

    def random_unitary(self):
        mat = stats.unitary_group.rvs(dim=int(2 ** self._qs), random_state=self.random_state)
        return mat

    def random_statevector(self):
        x = self.random_state.random(2 ** self._qs)
        x += x == 0
        x = -np.log(x)
        sumx = sum(x)
        phases = self.random_state.random(2 ** self._qs) * 2.0 * np.pi
        return np.sqrt(x / sumx) * np.exp(1j * phases)

    def Ginibre_Ensemble(self, nrow, ncol):
        ginibre = self.random_state.normal(size=(nrow, ncol)) + self.random_state.normal(size=(nrow, ncol)) * 1j
        return ginibre

    def HS(self, rank=None):
        mat = self.Ginibre_Ensemble(2 ** self._qs, rank)
        mat = mat.dot(mat.conj().T)
        return mat / np.trace(mat)

    def Bures(self, rank=None):
        density = np.eye(2 ** self._qs) + self.random_unitary()
        mat = density.dot(self.Ginibre_Ensemble(2 ** self._qs, rank))
        mat = mat.dot(mat.conj().T)
        return mat / np.trace(mat)

    def random_density_matrix(self, rank=None, method='Hilbert-Schmidt'):

        if rank is None:
            rank = 2 ** self._qs  # full rank

        if method == "Hilbert-Schmidt":
            rho = self.HS(rank)
        elif method == "Bures":
            rho = self.Bures(rank)
        else:
            raise NotImplemented(f"Error: method {method} is not implemented")

        return rho


class Haar_State:

    def __init__(self, qs, seed=None):
        self._qs = qs
        self.randomness = Randomness(qs=self._qs)
        self.seed = seed

    def pure_states(self, _):
        state = self.randomness.random_statevector()
        state_dm = np.outer(state, np.conj(state))
        return state_dm

    def sample_dm(self, n_size):  # K == D in equation (3) in the bias paper
        q_dm = list(map(self.pure_states, range(n_size)))
        q_dm = np.array(q_dm).reshape(n_size, 2 ** self._qs,
                                      2 ** self._qs)  # [self.n_size, 2 ** self._qs, 2 ** self._qs]
        return q_dm


class Hilbert_Schmidt:

    def __init__(self, qs, rank=None, seed=None):
        self._qs = qs
        self.randomness = Randomness(qs=self._qs)
        self.rank = rank
        self.seed = seed

    def hs_states(self, _):
        dm = self.randomness.random_density_matrix(rank=self.rank,
                                                   method='Hilbert-Schmidt')  # defualt is Hilbert-Schmidth
        return dm

    def sample_dm(self, n_size):
        hs_dm = list(map(self.hs_states, range(n_size)))
        hs_dm = np.array(hs_dm).reshape(n_size, 2 ** self._qs, 2 ** self._qs)
        return hs_dm


class Bures:

    def __init__(self, qs, rank=None):
        self._qs = qs
        self.randomness = Randomness(qs=self._qs)
        self.rank = rank

    def hs_states(self, _):
        dm = self.randomness.random_density_matrix(rank=self.rank, method='Bures')  # defualt is Hilbert-Schmidth
        return dm

    def sample_dm(self, n_size):
        hs_dm = list(map(self.hs_states, range(n_size)))
        hs_dm = np.array(hs_dm).reshape(n_size, 2 ** self._qs, 2 ** self._qs)
        return hs_dm


class eye_NN:

    def __init__(self, qs):
        self._qs = qs

    def I_states(self, _):
        state = np.identity(2 ** self._qs)
        return state

    def sample_dm(self, n_size):  # K == D in equation (3) in the bias paper
        q_dm = list(map(self.I_states, range(n_size)))
        q_dm = np.array(q_dm).reshape(n_size, 2 ** self._qs,
                                      2 ** self._qs)  # [self.n_size, 2 ** self._qs, 2 ** self._qs]
        return 1 / 4 * q_dm


class HS_Haar:

    def __init__(self, qs):
        self._qs = qs

    def sample_dm(self, n_size, Haar_to_HS=None):  # a fraction for Haar_to_HS. For eg. 10% --> 0.1
        haar_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size)
        hs_dm = Hilbert_Schmidt(qs=self._qs).sample_dm(n_size=n_size)
        if Haar_to_HS is None:
            a = np.random.uniform(low=0.0, high=1.0, size=[n_size, 1, 1])
        else:
            a = Haar_to_HS
        hs_haar_dm = (1 - a) * hs_dm + a * haar_dm
        return hs_haar_dm


class Mix_eye:

    def __init__(self, qs):
        self._qs = qs

    def sample_dm(self, n_size, eye_to_mix=None, states='HS'):  # a fraction for I_to_Mix. For eg. 10% --> 0.1
        if states == 'HS':
            mix_dm = Hilbert_Schmidt(qs=self._qs).sample_dm(n_size=n_size)
        if states == 'Haar':
            mix_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size)

        I_dm = eye_NN(qs=self._qs).sample_dm(n_size=n_size)
        if eye_to_mix is None:
            a = np.random.uniform(low=0.0, high=1.0, size=[n_size, 1, 1])
        else:
            a = eye_to_mix

        hs_haar_dm = (1 - a) * mix_dm + a * I_dm
        return hs_haar_dm


class Werner_Two_Q:

    def __init__(self,
                 qs=2, seed=None):
        self._qs = qs
        self.dim = np.log2(qs).astype(int)
        self.seed = seed
        self.randomness = Randomness(qs=self.dim, seed=self.seed)

    def haar_unitary_tensor_product(self, _):
        u_haar = np.kron(self.randomness.random_unitary(), self.randomness.random_unitary())
        return u_haar

    def rho_werner(self, n_size):
        rho_psi_m = 1 / 2. * np.array([[0., 0., 0., 0.],
                                         [0., 1., -1., 0.],
                                         [0., -1., 1., 0.],
                                         [0., 0., 0., 0.]]).reshape(1, 4, 4)
        I = np.eye(4)
        eta = np.random.uniform(low=0, high=1, size=[n_size, 1, 1])
        frac = (1 - eta) / 4
        assert frac.ndim == 3, 'frac_com is not a valid rank 3 tensor'
        assert frac.shape[0] == n_size, f'frac_com 0 dim shape does not match with n_size {n_size}'
        rho_w = eta * rho_psi_m + frac * I
        return rho_w  # shape of [n_size, dim, dim]

    def sample_dm(self, n_size):  # rho_w should be a rank 3 tensor (n_size, dim, dim)
        rho_w = self.rho_werner(n_size=n_size)
        unitary_mat = list(map(self.haar_unitary_tensor_product, range(n_size)))
        unitary_mat_array = np.array(unitary_mat).reshape(-1, 2**self._qs, 2**self._qs)
        unitary_mat_dag = np.transpose(unitary_mat_array, axes=(0, 2, 1)).conjugate()
        sampled_dm = np.matmul(unitary_mat_array, np.matmul(rho_w, unitary_mat_dag))
        return sampled_dm

class Maximally_Entangled_Mixed_States:

    def __init__(self, gamma, qs=2, seed=None):  # gamma: Concurrence
        self.gamma = gamma
        self.dim = np.log2(qs).astype(int)
        self.seed = seed
        self.randomness = Randomness(self.dim, seed=self.seed)

    def g_gamma(self):
        if self.gamma >= 2 / 3.:
            return self.gamma / 2.
        elif self.gamma < 2 / 3:
            return 1 / 3.
        else:
            raise ValueError

    def concurrence_purity(self):
        concurrence = self.gamma
        g = self.g_gamma()
        purity = 1 - 0.5 * (4 * g * (2 - 3 * g) - self.gamma ** 2)
        return (concurrence, purity)

    def rho_mems(self):
        g = self.g_gamma()
        rho = [g, 0, 0, self.gamma / 2.,
               0, 1 - 2 * g, 0, 0,
               0, 0, 0, 0,
               self.gamma / 2, 0, 0, g]
        rho = np.array(rho).reshape(1, 4, 4)
        return rho

    def haar_unitary_tensor_product(self, _):
        u_haar = np.kron(self.randomness.random_unitary(), self.randomness.random_unitary())
        return u_haar

    def sample_dm(self, n_size):
        rho = self.rho_mems()
        u_haar = list(map(self.haar_unitary_tensor_product, range(n_size)))
        u_haar = np.array(u_haar).reshape(n_size, 4, 4)
        u_haar_dag = np.transpose(np.conj(u_haar), (0, 2, 1))
        rho_sampled = np.matmul(u_haar, np.matmul(rho, u_haar_dag))
        return rho_sampled


class Maximally_Entangled_Mixed_States_Concurrence_Range:

    def __init__(self, gamma=[0, 1], qs=2, seed=None):
        self._qs = qs
        self.gamma = gamma

    @staticmethod
    def rho_sample(gamma_value):
        mems = Maximally_Entangled_Mixed_States(gamma=gamma_value).sample_dm(n_size=1)
        return mems

    def sample_dm(self, n_size):
        g = np.random.uniform(self.gamma[0], self.gamma[1], n_size)
        rho_mems = list(map(self.rho_sample, g))
        rho_mems_reshaped = np.array(rho_mems).reshape(-1, 2 ** self._qs, 2 ** self._qs)
        return rho_mems_reshaped


class Separable_States:

    def __init__(self, qs, pure=True, seed=None):
        self._qs = qs
        self.dim = np.log2(qs).astype(int)
        self.purity = pure
        self.seed = seed
        self.randomness = Randomness(self.dim, seed=self.seed)

    def sep_state(self, _):
        if self.purity:
            states = np.kron(self.randomness.random_density_matrix(rank=1),
                             self.randomness.random_density_matrix(rank=1))
        else:
            states = np.kron(self.randomness.random_density_matrix(),
                             self.randomness.random_density_matrix())
        return states

    def sample_dm(self, n_size):
        rho_s = list(map(self.sep_state, range(n_size)))
        rho_array = np.array(rho_s).reshape(-1, 2 ** self._qs, 2 ** self._qs)
        return rho_array


class MaiAlquierDist_Symmetric:

    def __init__(self, qs=2, alpha=0.1):
        self.alpha = alpha
        self._qs = qs

    def sample_alpha(self, n_size=1000):
        alpha = np.repeat(self.alpha, [2 ** self._qs])
        dist = np.random.dirichlet(alpha, size=n_size)
        sampled = dist.reshape(n_size, 2 ** self._qs, 1, 1)
        return sampled

    def sample_dm(self, n_size):
        q_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size * 2 ** self._qs)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = q_dm.reshape(n_size, 2 ** self._qs, 2 ** self._qs, 2 ** self._qs)
        alphas = self.sample_alpha(n_size)
        ma_states_array = alphas * haar_dm
        ma_states = np.sum(ma_states_array, axis=1)
        return ma_states

class MaiAlquierDist_Asymmetric:
    def __init__(self,
                 qs=2,
                 k_params=None,
                 alpha=[0.1, 0.2, 0.3, 0.4]) -> object:
        self.alpha = alpha
        self._qs = qs
        self.D = 2 ** self._qs
        self.K = self.D
        if k_params is not None:
            self.K = k_params

    def sample_alpha(self, n_size=1000):
        if isinstance(self.alpha, np.ndarray):
            assert self.alpha.ndim == 2, '|The given alpha must be a rank 2 tensor.'
            sampled = np.random.dirichlet(self.alpha, 1)
            sampled = np.squeeze(sampled)
        else:
            sampled = np.random.dirichlet(self.alpha, n_size)# [n_size, self._qs]
        sampled = sampled.reshape(n_size, -1, 1, 1)
        return sampled

    def sample_dm(self, n_size):
        q_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size * self.K)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = q_dm.reshape(n_size, self.K, 2 ** self._qs, 2 ** self._qs)  # [n_size, self._qs,
        # self._qs, self._qs]
        alphas = self.sample_alpha(n_size)  # [n_size, self._qs, 1, 1]

        ma_states_array = alphas * haar_dm  # [n_size, self._qs, self._qs, self._qs]
        ma_states = np.sum(ma_states_array,
                                  axis=1)  # [n_size, self._qs --> traced out and dropped, self._qs, self._qs]
        # --> [n_size, self._qs, self._qs]
        return ma_states

