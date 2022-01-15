import _pickle as pkl
import os

import numpy as np
import pandas as pd
import qiskit.quantum_info as qi
import tensorflow as tf
import tensorflow_probability as tfp



class Werner_Two_Q:

    def __init__(self,
                 qs=tf.constant(2.)):
        self._qs = qs

    def haar_unitary_tensor_product(self, _):
        u_haar = qi.random_unitary(dims=2).tensor(qi.random_unitary(dims=2))
        u_haar = u_haar.data
        return u_haar

    def rho_werner(self, n_size):
        rho_psi_m = 1 / 2. * tf.reshape([[0., 0., 0., 0.],
                                         [0., 1., -1., 0.],
                                         [0., -1., 1., 0.],
                                         [0., 0., 0., 0.]], (1, 4, 4))
        rho_psi_m = tf.cast(rho_psi_m, tf.complex128)
        I = tf.cast(tf.reshape(tf.eye(4), [1, 4, 4]), tf.complex128)
        eta = tf.random.uniform([n_size, 1, 1], minval=0, maxval=1.0)
        frac = (1. - eta) / 4.
        frac_com = tf.cast(frac, tf.complex128)
        eta_com = tf.cast(eta, tf.complex128)
        tf.debugging.assert_equal(tf.rank(frac_com), 3, message='frac_com is not a valid rank 3 tensor')
        tf.debugging.assert_equal(tf.shape(frac_com)[0], n_size,
                                  message=f'frac_com 0 dim shape does not match with n_size {n_size}')
        rho_w = tf.add(tf.multiply(eta_com, rho_psi_m), tf.multiply(frac_com, I))
        return rho_w  # shape of [n_size, dim, dim]

    def sample_dm(self, n_size):  # rho_w should be a rank 3 tensor (n_size, dim, dim)
        rho_w = self.rho_werner(n_size=n_size)
        unitary_mat = list(map(self.haar_unitary_tensor_product, range(n_size)))
        unitary_mat_array = tf.reshape(tf.cast(unitary_mat, tf.complex128), [-1, 2 ** self._qs, 2 ** self._qs])
        unitary_mat_array_dag = tf.transpose(unitary_mat_array, perm=[0, 2, 1], conjugate=True)
        sampled_dm = tf.matmul(unitary_mat_array, tf.matmul(rho_w, unitary_mat_array_dag))
        return sampled_dm


class MaiAlquierDist_BME:

    def __init__(self,
                 qs=tf.TensorSpec(shape=1, dtype=tf.int64),
                 alpha=tf.TensorSpec(shape=1, dtype=tf.float32)) -> object:
        self.alpha = alpha
        self._qs = qs

    @staticmethod
    def _cast_complex(x):
        return tf.cast(x, tf.complex128)

    @tf.function
    def sample_dm(self, n_size=tf.TensorSpec(shape=1, dtype=tf.int64)):
        self.n_size = n_size
        x = tf.random.normal([self.n_size, 2 * 2 ** self._qs * 2 ** self._qs], 0., 1.)
        Xr = tf.reshape(x[:, :2 ** self._qs * 2 ** self._qs], [self.n_size, 2 ** self._qs, 2 ** self._qs])
        Xi = tf.reshape(x[:, 2 ** self._qs * 2 ** self._qs:], [self.n_size, 2 ** self._qs, 2 ** self._qs])
        Xr = self._cast_complex(Xr)
        Xi = self._cast_complex(Xi)
        X = Xr + 1j * Xi
        W = X / tf.expand_dims(tf.norm(X, axis=1), axis=1)
        print('shape of W', W.shape)
        if isinstance(self.alpha, float):
            gamma_factor = self._cast_complex(tf.random.gamma([self.n_size, 2 ** self._qs], alpha=self.alpha, beta=1.))
        else:
            g_tensor = tf.vectorized_map(lambda x: tf.random.gamma([2 ** self._qs], x), self.alpha)
            gamma_factor = self._cast_complex(tf.reshape(g_tensor, [-1, 2 ** self._qs]))
        gamma_factor_norm = gamma_factor / tf.expand_dims(tf.reduce_sum(gamma_factor, axis=1), axis=1)
        gama_diag_batch = tf.vectorized_map(lambda x: tf.linalg.diag(x), gamma_factor_norm)  # rank 3 tensors
        rho = tf.linalg.matmul(W, tf.linalg.matmul(gama_diag_batch, W, adjoint_b=True))

        return rho


class Haar_NN:

    def __init__(self, qs):
        self._qs = qs

    def pure_states(self, _):
        state = qi.random_statevector(dims=2 ** self._qs)
        state_dm = state.to_operator()
        state_np = state_dm.data
        return state_np

    def sample_dm(self, n_size):  # K == D in equation (3) in the bias paper
        q_dm = list(map(self.pure_states, range(n_size)))
        q_dm = np.array(q_dm).reshape(n_size, 2 ** self._qs,
                                      2 ** self._qs)  # [self.n_size, 2 ** self._qs, 2 ** self._qs]
        return q_dm


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


class HS_NN:

    def __init__(self, qs):
        self._qs = qs

    def hs_states(self, _):
        dm = qi.random_density_matrix(dims=2 ** self._qs)  # defualt is Hilbert-Schmidth
        dm_np = dm.data
        return dm_np

    def sample_dm(self, n_size):
        hs_dm = list(map(self.hs_states, range(n_size)))
        hs_dm = np.array(hs_dm).reshape(n_size, 2 ** self._qs, 2 ** self._qs)
        return hs_dm


class Bures_NN:

    def __init__(self, qs):
        self._qs = qs

    def hs_states(self, _):
        dm = qi.random_density_matrix(dims=2 ** self._qs, method='Bures')  # defualt is Hilbert-Schmidth
        dm_np = dm.data
        return dm_np

    def sample_dm(self, n_size):
        hs_dm = list(map(self.hs_states, range(n_size)))
        hs_dm = np.array(hs_dm).reshape(n_size, 2 ** self._qs, 2 ** self._qs)
        return hs_dm

class HS_Haar_NN:

    def __init__(self, qs):
        self._qs = qs

    def sample_dm(self, n_size, Haar_to_HS=None):  # a fraction for Haar_to_HS. For eg. 10% --> 0.1
        haar_dm = Haar_NN(qs=self._qs).sample_dm(n_size=n_size)
        hs_dm = HS_NN(qs=self._qs).sample_dm(n_size=n_size)
        if Haar_to_HS is None:
            a = np.random.uniform(low=0.0, high=1.0, size=[n_size, 1, 1])
        else:
            a = Haar_to_HS
        hs_haar_dm = (1 - a) * hs_dm + a * haar_dm
        return hs_haar_dm


class Mix_eye_NN:

    def __init__(self, qs):
        self._qs = qs

    def sample_dm(self, n_size, eye_to_mix=None, states='HS'):  # a fraction for I_to_Mix. For eg. 10% --> 0.1
        if states == 'HS':
            mix_dm = HS_NN(qs=self._qs).sample_dm(n_size=n_size)
        if states == 'Haar':
            mix_dm = Haar_NN(qs=self._qs).sample_dm(n_size=n_size)

        I_dm = eye_NN(qs=self._qs).sample_dm(n_size=n_size)
        if eye_to_mix is None:
            a = np.random.uniform(low=0.0, high=1.0, size=[n_size, 1, 1])
        else:
            a = eye_to_mix

        hs_haar_dm = (1 - a) * mix_dm + a * I_dm
        return hs_haar_dm


class MaiAlquierDist_Symmetric_NN:

    def __init__(self,
                 qs=2,
                 alpha=tf.TensorSpec(shape=1, dtype=tf.float32)) -> object:
        self.alpha = alpha
        self._qs = qs

    @staticmethod
    def _cast_complex(x):
        return tf.cast(x, tf.complex128)

    def sample_alpha(self, n_size=tf.TensorSpec(shape=1, dtype=tf.int64)):
        alpha = tf.repeat(self.alpha, [2 ** self._qs])
        dist = tfp.distributions.Dirichlet(alpha)
        sampled = dist.sample(n_size)  # [n_size, self._qs]
        sampled = tf.expand_dims(sampled, axis=-1)  # [n_size, self._qs, 1]
        sampled = tf.expand_dims(sampled, axis=-1)  # [n_size, self._qs, 1, 1]
        return sampled

    def sample_dm(self, n_size):
        q_dm = Haar_NN(qs=self._qs).sample_dm(n_size=n_size * 2 ** self._qs)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = tf.reshape(q_dm, [n_size, 2 ** self._qs, 2 ** self._qs, 2 ** self._qs])  # [n_size, self._qs,
        # self._qs, self._qs]
        alphas = self.sample_alpha(n_size)  # [n_size, self._qs, 1, 1]
        alphas = self._cast_complex(alphas)
        ma_states_array = tf.multiply(alphas, haar_dm)  # [n_size, self._qs, self._qs, self._qs]
        ma_states = tf.reduce_sum(ma_states_array,
                                  axis=1)  # [n_size, self._qs --> traced out and dropped, self._qs, self._qs]
        # --> [n_size, self._qs, self._qs]
        return ma_states


class MaiAlquierDist_Asymmetric_NN:

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

    @staticmethod
    def _cast_complex(x):
        return tf.cast(x, tf.complex128)

    def sample_alpha(self, n_size=tf.TensorSpec(shape=1, dtype=tf.int64)):
        # if purity is not None:
        #     self.alpha = self.D * (1 - purity) / (self.D * (purity * self.D - 2) + 1)
        # alpha = tf.repeat(self.alpha, [2 ** self._qs])
        dist = tfp.distributions.Dirichlet(self.alpha)
        if isinstance(self.alpha, np.ndarray):
            tf.debugging.assert_equal(self.alpha.ndim, 2, '|The given alpha must be a rank 2 tensor.')
            sampled = dist.sample(1)
            sampled = tf.squeeze(sampled)
        else:
            sampled = dist.sample(n_size)  # [n_size, self._qs]
        sampled = tf.expand_dims(sampled, axis=-1)  # [n_size, self._qs, 1]
        sampled = tf.expand_dims(sampled, axis=-1)  # [n_size, self._qs, 1, 1]
        return sampled

    def sample_dm(self, n_size):
        q_dm = Haar_NN(qs=self._qs).sample_dm(n_size=n_size * self.K)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = tf.reshape(q_dm, [n_size, self.K, 2 ** self._qs, 2 ** self._qs])  # [n_size, self._qs,
        # self._qs, self._qs]
        alphas = self.sample_alpha(n_size)  # [n_size, self._qs, 1, 1]
        alphas = self._cast_complex(alphas)
        ma_states_array = tf.multiply(alphas, haar_dm)  # [n_size, self._qs, self._qs, self._qs]
        ma_states = tf.reduce_sum(ma_states_array,
                                  axis=1)  # [n_size, self._qs --> traced out and dropped, self._qs, self._qs]
        # --> [n_size, self._qs, self._qs]
        return ma_states


class Measurements_Ideal:

    def __init__(self, qs=2):
        self._qs = qs
        self.projectors = pd.read_pickle(f'./utils/ibm_projectors_array_qs_{self._qs}.pickle')

    def get_tau_cholesky(self, density_matrix):
        try:
            chol = np.linalg.cholesky(density_matrix)
        except:
            dm = (1. - 0.0000001) * density_matrix + 0.0000001 / 4 * np.eye(2 ** self._qs)
            chol = np.linalg.cholesky(dm)
            pass
        tau_first = np.real(chol.diagonal())  # len 16 for 4 qubits
        ele_off_diags = chol[np.tril_indices(n=2 ** self._qs,
                                             k=-1)]  # https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html
        tau_second = []
        for i in ele_off_diags:
            re, img = np.real(i), np.imag(i)
            tau_second.append(re)
            tau_second.append(img)
        tau = np.concatenate([tau_first, tau_second])
        return tau

    def get_tomo_from_density_matrix(self, density_matrix, projectors):
        tomo = np.real(np.trace(np.matmul(density_matrix, projectors), axis1=1, axis2=2))
        return tomo

    def tomography_data(self, density_matrix, save_file=False, filename='pickle.pickle'):
        if not isinstance(density_matrix, np.ndarray):
            density_matrix = density_matrix.numpy()
        tomo_list = list(map(lambda x: self.get_tomo_from_density_matrix(x, self.projectors), density_matrix))
        tomo_array = np.array(tomo_list).reshape(-1, 6 ** self._qs)
        tau_list = list(map(self.get_tau_cholesky, density_matrix))
        tau_array = np.array(tau_list).reshape(-1, 2 ** self._qs * 2 ** self._qs)

        if save_file:
            if not os.path.exists('./data'):
                os.mkdir('./data')
                tf.print('./data folder has been created and ')
            with open(f'./data/{filename}', 'wb') as f:
                tf.print(f'tomography data has been saved into ./data/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix], f, -1)

        return [tomo_array, tau_array]


class Measurements_Random:

    def __init__(self, qs=2, n_meas=1024):
        self._qs = qs
        self.n_shots = n_meas
        self.projectors = pd.read_pickle(f'./utils/ibm_projectors_array_qs_{self._qs}.pickle')
        self.n_proj = self.projectors.reshape(3 ** self._qs, self._qs ** 2, self._qs ** 2, self._qs ** 2)
        self.proj_used_rank_list = []

    def get_tau_cholesky(self, density_matrix):
        try:
            chol = np.linalg.cholesky(density_matrix)
        except:
            dm = (1. - 0.0000001) * density_matrix + 0.0000001 / 4 * np.eye(2 ** self._qs)
            chol = np.linalg.cholesky(dm)
            pass
        tau_first = np.real(chol.diagonal())  # len 16 for 4 qubits
        ele_off_diags = chol[np.tril_indices(n=2 ** self._qs,
                                             k=-1)]  # https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html
        tau_second = []
        for i in ele_off_diags:
            re, img = np.real(i), np.imag(i)
            tau_second.append(re)
            tau_second.append(img)
        tau = np.concatenate([tau_first, tau_second])
        return tau

    def get_tomo_from_density_matrix(self, projectors):
        tomo = np.trace(np.real(np.matmul(self.rho, projectors))).astype(np.float32)  # one by one multiplication
        return tomo

    def measure_specific_basis(self, proj_int, dm_one):
        self.rho = dm_one
        # n_proj = self.projectors.reshape(3 ** self._qs, self._qs ** 2, self._qs ** 2, self._qs ** 2)
        projectors = self.n_proj[proj_int]  # --> [4, 4, 4]
        tomo_probs = list(map(self.get_tomo_from_density_matrix, projectors))
        tomo_probs = np.array(tomo_probs).flatten()
        draw = np.random.multinomial(1, tomo_probs)
        arg_max_rank = np.argmax(draw)
        measurements = np.zeros(6 ** self._qs).astype(np.float32)
        measurements[4 * proj_int + arg_max_rank] = 1.
        self.proj_used_rank_list.append(4 * proj_int + arg_max_rank)
        return measurements

    def measurement_array(self, dm=None):
        dm = np.squeeze(dm)
        proj_rank = np.random.randint(0, 9, self.n_shots)
        measurement_list = list(map(lambda x: self.measure_specific_basis(x, dm), proj_rank))
        measurement_array = np.array(measurement_list).reshape(-1, 6 ** self._qs)
        self.count += 1
        if self.count % 1000 == 0:
            print(f'dm_finished: {self.count}')
        return measurement_array

    def tomography_data(self, density_matrix, norm=True, save_file=False, filename='pickle.pickle'):
        self.count = 1
        measurements = list(map(self.measurement_array, density_matrix))
        measurements = np.array(measurements).reshape(-1, 9 * self.n_shots, 6 ** self._qs)
        if norm:
            tomo_array = np.sum(measurements, axis=1) / self.n_shots
        else:
            tomo_array = np.sum(measurements, axis=1)
        tau_list = list(map(self.get_tau_cholesky, density_matrix))
        tau_array = np.array(tau_list).reshape(-1, 2 ** self._qs * 2 ** self._qs)

        if save_file:
            if not os.path.exists('./data/data_shots'):
                os.mkdir('./data/data_shots')
                tf.print('./data/data_shots folder has been created and ')
            with open(f'./data/data_shots/{filename}', 'wb') as f:
                tf.print(f'tomography data has been saved into ./data/data_shots/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix, self.proj_used_rank_list], f, -1)

        # return measurements, self.proj_used_rank_list
        return tomo_array, tau_array


class Measurements_Shots_IBMQ:

    def __init__(self, qs=2, shots=1024):
        self._qs = qs
        self.n_shots = shots
        self.projectors = pd.read_pickle(f'./utils/ibm_projectors_array_qs_{self._qs}.pickle')
        # self.n_proj = self.projectors.reshape(3 ** self._qs, self._qs ** 2, self._qs ** 2, self._qs ** 2)
        self.proj_used_rank_list = []

    def get_tau_cholesky(self, density_matrix):
        try:
            chol = np.linalg.cholesky(density_matrix)
        except:
            dm = (1. - 0.0000001) * density_matrix + 0.0000001 / 4 * np.eye(2 ** self._qs)
            chol = np.linalg.cholesky(dm)
            pass
        tau_first = np.real(chol.diagonal())  # len 16 for 4 qubits
        ele_off_diags = chol[np.tril_indices(n=2 ** self._qs,
                                             k=-1)]  # https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html
        tau_second = []
        for i in ele_off_diags:
            re, img = np.real(i), np.imag(i)
            tau_second.append(re)
            tau_second.append(img)
        tau = np.concatenate([tau_first, tau_second])
        return tau

    def get_tomo_from_density_matrix(self, rho, projectors):  # --> Ideal measurements
        tomo = np.trace(np.real(np.matmul(rho, projectors)), axis1=1, axis2=2).astype(
            np.float32)  # multiplication with broadcasting the arrays
        return tomo

    def draw_multinomial(self, row_prob):
        return np.random.multinomial(self.n_shots, row_prob)

    def measurement_array(self, dm=None):
        n_proj = self.projectors.reshape(-1, 4, 4)
        dm = dm.reshape(1, 4, 4)
        ideal_tomo_list = self.get_tomo_from_density_matrix(dm, n_proj)
        ideal_tomo_array = np.array(ideal_tomo_list).reshape(-1, 2 ** self._qs)
        measurement_list = list(map(self.draw_multinomial, ideal_tomo_array))
        measurement_array = np.array(measurement_list).flatten()
        self.count += 1
        if self.count % 1000 == 0:
            print(f'dm_finished: {self.count}')
        return measurement_array

    def tomography_data(self, density_matrix, norm=True, save_file=False, filename='pickle.pickle'):
        self.count = 1
        measurements = list(map(self.measurement_array, density_matrix))
        measurements = np.array(measurements).reshape(-1, 6 ** self._qs)
        if norm:
            tomo_array = measurements / self.n_shots
        else:
            tomo_array = measurements
        tau_list = list(map(self.get_tau_cholesky, density_matrix))
        tau_array = np.array(tau_list).reshape(-1, 2 ** self._qs * 2 ** self._qs)

        if save_file:
            if not os.path.exists('./data/data_shots'):
                os.mkdir('./data/data_shots')
                tf.print('./data/data_shots folder has been created and ')
            with open(f'./data/data_shots/{filename}', 'wb') as f:
                tf.print(f'tomography data has been saved into ./data/data_shots/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix, self.proj_used_rank_list], f, -1)

        # return measurements, self.proj_used_rank_list
        return tomo_array, tau_array

# from source import DataSimulator as data_sim

# x = Werner_Two_Q().sample_dm(n_size=10000)
# print(x)
# %%
