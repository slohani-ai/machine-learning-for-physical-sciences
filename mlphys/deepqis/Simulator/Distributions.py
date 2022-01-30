import numpy as np
import qiskit.quantum_info as qi
import tensorflow as tf
import tensorflow_probability as tfp


class Haar_State:

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


class Hilbert_Schmidt:

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


class Bures:

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


class MaiAlquierDist_Symmetric:

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

    def sample_dm(self, n_size, numpy_array=True):
        q_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size * 2 ** self._qs)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = tf.reshape(q_dm, [n_size, 2 ** self._qs, 2 ** self._qs, 2 ** self._qs])  # [n_size, self._qs,
        # self._qs, self._qs]
        alphas = self.sample_alpha(n_size)  # [n_size, self._qs, 1, 1]
        alphas = self._cast_complex(alphas)
        ma_states_array = tf.multiply(alphas, haar_dm)  # [n_size, self._qs, self._qs, self._qs]
        ma_states = tf.reduce_sum(ma_states_array,
                                  axis=1)  # [n_size, self._qs --> traced out and dropped, self._qs, self._qs]
        # --> [n_size, self._qs, self._qs]
        if numpy_array:
            ma_states = ma_states.numpy()
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

    def sample_dm(self, n_size, numpy_array=True):
        q_dm = Haar_State(qs=self._qs).sample_dm(n_size=n_size * self.K)  # [self.n_size * 2**self._qs,
        # 2 ** self._qs, 2 ** self._qs]
        haar_dm = tf.reshape(q_dm, [n_size, self.K, 2 ** self._qs, 2 ** self._qs])  # [n_size, self._qs,
        # self._qs, self._qs]
        alphas = self.sample_alpha(n_size)  # [n_size, self._qs, 1, 1]
        alphas = self._cast_complex(alphas)
        ma_states_array = tf.multiply(alphas, haar_dm)  # [n_size, self._qs, self._qs, self._qs]
        ma_states = tf.reduce_sum(ma_states_array,
                                  axis=1)  # [n_size, self._qs --> traced out and dropped, self._qs, self._qs]
        # --> [n_size, self._qs, self._qs]
        if numpy_array:
            ma_states = ma_states.numpy()
        return ma_states


class MaiAlquierDist_Gamma:

    def __init__(self,
                 qs=tf.TensorSpec(shape=1, dtype=tf.int64),
                 alpha=tf.TensorSpec(shape=1, dtype=tf.float32)) -> object:
        self.alpha = alpha
        self._qs = qs

    @staticmethod
    def _cast_complex(x):
        return tf.cast(x, tf.complex128)

    @tf.function
    def sample_dm(self, n_size=tf.TensorSpec(shape=1, dtype=tf.int64), numpy_array=False):
        self.n_size = n_size
        x = tf.random.normal([self.n_size, 2 * 2 ** self._qs * 2 ** self._qs], 0., 1.)
        Xr = tf.reshape(x[:, :2 ** self._qs * 2 ** self._qs], [self.n_size, 2 ** self._qs, 2 ** self._qs])
        Xi = tf.reshape(x[:, 2 ** self._qs * 2 ** self._qs:], [self.n_size, 2 ** self._qs, 2 ** self._qs])
        Xr = self._cast_complex(Xr)
        Xi = self._cast_complex(Xi)
        X = Xr + 1j * Xi
        W = X / tf.expand_dims(tf.norm(X, axis=1), axis=1)
        # print('shape of W', W.shape)
        if isinstance(self.alpha, float):
            gamma_factor = self._cast_complex(tf.random.gamma([self.n_size, 2 ** self._qs], alpha=self.alpha, beta=1.))
        else:
            g_tensor = tf.vectorized_map(lambda x: tf.random.gamma([2 ** self._qs], x), self.alpha)
            gamma_factor = self._cast_complex(tf.reshape(g_tensor, [-1, 2 ** self._qs]))
        gamma_factor_norm = gamma_factor / tf.expand_dims(tf.reduce_sum(gamma_factor, axis=1), axis=1)
        gama_diag_batch = tf.vectorized_map(lambda x: tf.linalg.diag(x), gamma_factor_norm)  # rank 3 tensors
        rho = tf.linalg.matmul(W, tf.linalg.matmul(gama_diag_batch, W, adjoint_b=True))
        return rho
