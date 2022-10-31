import _pickle as pkl
import itertools
import os
import sys

import numpy as np
import pandas as pd

__author__ = 'Sanjaya Lohani'
__email__ = 'slohani@mlphys.com'
__licence__ = 'Apache 2.0'
__website__ = "sanjayalohani.com"

class Gen_Basis_Order:

    def __init__(self, qs):
        self._qs = qs

    def Povm_List_Qubit_1(self, standard_list=['d', 'a', 'r', 'l', 'h', 'v']):

        return standard_list

    def Povm_List_Qubit_2(self):
        standard_list = self.Povm_List_Qubit_1()
        iter_for = [['d', 'a'], ['r', 'l'], ['h', 'v']]
        povm_list = []
        for j in iter_for:
            povm_list.append(list(itertools.product(standard_list, j)))
        p_list = []
        for i in range(len(povm_list)):
            p_list.append([list(j) for j in povm_list[i]])

        return p_list

    def Povm_List_Qubit_3(self):
        dd, rr, hh = self.Povm_List_Qubit_2()
        iter_for = [['d', 'a'], ['r', 'l'], ['h', 'v']]
        povm_list = []
        for j in iter_for:
            for k in [dd, rr, hh]:
                povm_list.append(list(itertools.product(k, j)))
        p_list = []
        for i in range(len(povm_list)):
            p_list.append([list(j) for j in povm_list[i]])

        return p_list

    def Povm_List_Qubit_4(self):
        dd_2, dd_3, dd_4, rr_2, rr_3, rr_4, hh_2, hh_3, hh_4 = self.Povm_List_Qubit_3()
        iter_for = [['d', 'a'], ['r', 'l'], ['h', 'v']]
        povm_list = []
        for j in iter_for:
            for k in [dd_2, dd_3, dd_4, rr_2, rr_3, rr_4, hh_2, hh_3, hh_4]:
                povm_list.append(list(itertools.product(k, j)))
        p_list = []
        for i in range(len(povm_list)):
            p_list.append([list(j) for j in povm_list[i]])
        return p_list

    def Convert_to_Projections(self, raw_povm_list, qs=3):
        pp_list = []
        for i in raw_povm_list:
            for k in i:
                un_list = list(itertools.chain.from_iterable(k))
                pp_list.append(un_list)
        if qs > 3:
            ppp_list = []
            for i in pp_list:
                un_list = list(itertools.chain.from_iterable(i))
                ppp_list.append(un_list)
            pp_list = ppp_list
        return pp_list

    def Basis_Order(self):
        if self._qs == 1:
            Plist = self.Povm_List_Qubit_1()
        elif self._qs == 2:
            Plist = self.Povm_List_Qubit_2()
        elif self._qs == 3:
            Plist = self.Povm_List_Qubit_3()
        elif self._qs == 4:
            Plist = self.Povm_List_Qubit_4()
        else:
            sys.exit('Qubit size must be less or equal to 4.')
        converted = self.Convert_to_Projections(Plist, qs=4)
        titles = [''.join(i) for i in converted]
        return titles


class Gen_General_Basis_Order:

    def __init__(self, qs):
        self._qs = qs
        self._standard_basis = [['d', 'a'], ['r', 'l'], ['h', 'v']]

    @property
    def POVM_List_Qubit_1(self):
        return np.concatenate(self._standard_basis).tolist()

    @property
    def POVM_List_Qubit_2(self):
        proj_order_pair = list(itertools.product(self._standard_basis, repeat=self._qs))
        proj_list = []
        for j in range(len(proj_order_pair)):
            proj_list.append(list(itertools.product(proj_order_pair[j][0], proj_order_pair[j][1])))
        proj_final_list = [''.join(i) for i in np.concatenate(proj_list)]
        return proj_final_list

    @property
    def POVM_List_Qubit_3(self):
        proj_order_pair = list(itertools.product(self._standard_basis, repeat=self._qs))
        proj_list = []
        for j in range(len(proj_order_pair)):
            proj_list.append(list(itertools.product(proj_order_pair[j][0], proj_order_pair[j][1], proj_order_pair[j][2])))
        proj_final_list = [''.join(i) for i in np.concatenate(proj_list)]
        return proj_final_list

    @property
    def POVM_List_Qubit_4(self):
        proj_order_pair = list(itertools.product(self._standard_basis, repeat=self._qs))
        proj_list = []
        for j in range(len(proj_order_pair)):
            proj_list.append(list(itertools.product(proj_order_pair[j][0], proj_order_pair[j][1],
                                                    proj_order_pair[j][2], proj_order_pair[j][3])))
        proj_final_list = [''.join(i) for i in np.concatenate(proj_list)]
        return proj_final_list

    def Basis_Order(self):

        if self._qs == 1:
            Plist = self.POVM_List_Qubit_1
        elif self._qs == 2:
            Plist = self.POVM_List_Qubit_2
        elif self._qs == 3:
            Plist = self.POVM_List_Qubit_3
        elif self._qs == 4:
            Plist = self.POVM_List_Qubit_4
        else:
            raise ValueError('This version only supports upto 4 qubits.')
        return Plist


class BellPair:
    def __init__(self):
        self.rho_phi_p = 1 / 2. * np.array([[1., 0., 0., 1.],
                                              [0., 0., 0., 0.],
                                              [0., 0, 0, 0],
                                              [1., 0., 0., 1.]]).reshape(4, 4)

        self.rho_phi_m = 1 / 2. * np.array([[1., 0., 0., -1.],
                                              [0., 0., 0., 0.],
                                              [0., 0., 0., 0.],
                                              [-1., 0., 0., 1.]]).reshape(4, 4)

        self.rho_psi_p = 1 / 2. * np.array([[0., 0., 0., 0.],
                                              [0., 1., 1., 0.],
                                              [0., 1., 1., 0.],
                                              [0., 0., 0., 0.]]).reshape(4, 4)

        self.rho_psi_m = 1 / 2. * np.array([[0., 0., 0., 0.],
                                              [0., 1., -1., 0.],
                                              [0., -1., 1., 0.],
                                              [0., 0., 0., 0.]]).reshape(4, 4)

    def __call__(self, *args, **kwargs):
        bell_pair = np.stack((self.rho_phi_p, self.rho_phi_m, self.rho_psi_p, self.rho_phi_m), axis=0)
        return bell_pair


class MultiQubitSystem:

    def __init__(self, qs=4):
        self._qs = qs
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
        ibmq_proj = Gen_Basis_Order(qs=self._qs).Basis_Order()
        # print(ibmq_proj)
        Proj = list(map(self.Kron_Povm, ibmq_proj))
        Proj = np.array(Proj).reshape(-1, 2 ** self._qs, 2 ** self._qs)
        return Proj

    def General_Scheme_Projectors(self):
        # assert (self._qs == 2), "This currently only supports two-qubits system."
        # proj_list = ['dd', 'da', 'ad', 'aa', 'dr', 'dl',
        #              'ar', 'al', 'dh', 'dv', 'ah', 'av',
        #              'rd', 'ra', 'ld', 'la', 'rr', 'rl',
        #              'lr', 'll', 'rh', 'rv', 'lh', 'lv',
        #              'hd', 'ha', 'vd', 'va', 'hr', 'hl',
        #              'vr', 'vl', 'hh', 'hv', 'vh', 'vv']
        proj_list = Gen_General_Basis_Order(qs=self._qs).Basis_Order()
        proj_list_map = list(map(self.Kron_Povm, proj_list))
        proj_array = np.array(proj_list_map).reshape(-1, 2 ** self._qs, 2 ** self._qs)
        return proj_array


class Ideal_Measurements:

    def __init__(self, qs=2, basis_set='general'):
        self._qs = qs
        if not os.path.exists(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle'):
            print(f'| To accelerate the simulation, a new projector_{basis_set} folder is created.')
            try:
                os.mkdir(f'projectors_{basis_set}')
            except:
                print(f'|Found projectors_{basis_set} folder! Appending some files ...')
            mqs = MultiQubitSystem(qs=self._qs)
            if basis_set == 'general':
                proj = mqs.General_Scheme_Projectors()
            elif basis_set == 'nisq':
                proj = mqs.NISQ_Projectors()
            else:
                raise ValueError('basis_set either be general or nisq.')
            # proj = mqs.NISQ_Projectors()
            with open(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle', 'wb') as f:
                pkl.dump(proj, f, -1)
            self.projectors = proj
        else:
            self.projectors = pd.read_pickle(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle')

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
                print('./data folder has been created and ')
            with open(f'./data/{filename}', 'wb') as f:
                print(f'tomography data has been saved into ./data/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix], f, -1)

        return [tomo_array, tau_array]


class Random_Measurements:

    ''' This is same as the Random_Measurements_Old but is much more faster.'''

    def __init__(self, qs=2, n_meas=1024, basis_set='general'):
        self._qs = qs
        self.n_shots = n_meas

        if not os.path.exists(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle'):
            print(
                f'| To accelerate the simulation, General Scheme Projector file is created in a new projectors_{basis_set} folder.')

            mqs = MultiQubitSystem(qs=self._qs)
            if basis_set == 'general':
                proj = mqs.General_Scheme_Projectors()
            elif basis_set == 'nisq':
                proj = mqs.NISQ_Projectors()
            else:
                raise ValueError('basis_set either be general or nisq.')
            try:
                os.mkdir(f'projectors_{basis_set}')
            except:
                print(f'|Found projectors_{basis_set} folder! Appending some files ...')

            with open(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}_general_scheme.pickle', 'wb') as f:
                pkl.dump(proj, f, -1)
            self.projectors = proj
        else:
            self.projectors = pd.read_pickle(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle')

        self.n_proj = self.projectors.reshape(3 ** self._qs, 2**self._qs , 2**self._qs, 2**self._qs)
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

    def get_tomo_from_density_matrix(self, density_matrix, projectors):

        tomo = np.real(np.trace(np.matmul(density_matrix, projectors), axis1=1, axis2=2))
        return tomo

    def measurement_array(self, dm=None):
        dm = dm.reshape(1, 2**self._qs, 2**self._qs)
        tomo_ideal = self.get_tomo_from_density_matrix(density_matrix=dm, projectors=self.projectors)
        tomo_ideal_array = tomo_ideal.reshape(3**self._qs, 2**self._qs)
        proj_rank = np.random.randint(0, 3**self._qs, self.n_shots)
        occurance_rank = [np.count_nonzero(proj_rank==i) for i in range(3**self._qs)]
        tomo_list = [np.random.multinomial(occurance_rank[i], tomo_ideal_array[i]) for i in range(3**self._qs)]
        tomo_array = np.concatenate(tomo_list)
        return tomo_array

    def tomography_data(self, density_matrix, norm=True, save_file=False, filename='pickle.pickle'):
        self.count = 1
        measurements = list(map(self.measurement_array, density_matrix))
        measurements = np.array(measurements).reshape(density_matrix.shape[0], 6 ** self._qs)
        if norm:
            tomo_array = measurements / self.n_shots
        else:
            tomo_array = measurements
        tau_list = list(map(self.get_tau_cholesky, density_matrix))
        tau_array = np.array(tau_list).reshape(-1, 2 ** self._qs * 2 ** self._qs)

        if save_file:
            if not os.path.exists('./data_measurements'):
                os.mkdir('./data_measurements')
                print('./data_measurements folder has been created and ')
            with open(f'./data_measurements/{filename}', 'wb') as f:
                print(f'tomography data has been saved into ./data_measurements/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix, self.proj_used_rank_list], f, -1)

        # return measurements, self.proj_used_rank_list
        return tomo_array, tau_array


class NISQ_Shots:

    def __init__(self, qs=2, shots=1024, basis_set='nisq'):
        self._qs = qs
        self.n_shots = shots

        if not os.path.exists(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle'):
            print(f'| To accelerate the simulation, Projectors_{basis_set} file is created in a new projectors folder.')

            mqs = MultiQubitSystem(qs=self._qs)
            if basis_set == 'general':
                proj = mqs.General_Scheme_Projectors()
            elif basis_set == 'nisq':
                proj = mqs.NISQ_Projectors()
            else:
                raise ValueError('basis_set either be general or nisq.')
            # proj = mqs.NISQ_Projectors()
            try:
                os.mkdir(f'projectors_{basis_set}')
            except:
                print(f'|Found projectors_{basis_set} folder! Appending some files ...')

            with open(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle', 'wb') as f:
                pkl.dump(proj, f, -1)
            self.projectors = proj
        else:
            self.projectors = pd.read_pickle(f'./projectors_{basis_set}/projectors_array_qs_{self._qs}.pickle')
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
        n_proj = self.projectors.reshape(-1, 2**self._qs, 2**self._qs)
        dm = dm.reshape(1, 2**self._qs, 2**self._qs)
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
                print('./data/data_shots folder has been created and ')
            with open(f'./data/data_shots/{filename}', 'wb') as f:
                print(f'tomography data has been saved into ./data/data_shots/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix, self.proj_used_rank_list], f, -1)

        # return measurements, self.proj_used_rank_list
        return tomo_array, tau_array

class Random_Measurements_Old:

    def __init__(self, qs=2, n_meas=1024, basis_set='general'):
        self._qs = qs
        self.n_shots = n_meas

        if not os.path.exists(f'./projectors/projectors_array_qs_{self._qs}_general_scheme.pickle'):
            print(
                '| To accelerate the simulation, General Scheme Projector file is created in a new projectors folder.')
            mqs = MultiQubitSystem(qs=self._qs)
            if basis_set == 'general':
                proj = mqs.General_Scheme_Projectors()
            elif basis_set == 'nisq':
                proj = mqs.NISQ_Projectors()
            else:
                raise ValueError('basis_set either be general or nisq.')
            try:
                os.mkdir('projectors')
            except:
                print('|Found projectors folder! Appending some files ...')

            with open(f'projectors_array_qs_{self._qs}_general_scheme.pickle', 'wb') as f:
                pkl.dump(proj, f, -1)
            self.projectors = proj
        else:
            self.projectors = pd.read_pickle(f'./projectors/projectors_array_qs_{self._qs}_general_scheme.pickle')

        self.n_proj = self.projectors.reshape(3 ** self._qs, 2**self._qs , 2**self._qs, 2**self._qs)
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

    def measure_specific_basis(self, proj_int, dm_one, proj_rank_list=False):
        self.rho = dm_one
        # n_proj = self.projectors.reshape(3 ** self._qs, self._qs ** 2, self._qs ** 2, self._qs ** 2)
        projectors = self.n_proj[proj_int]  # --> [4, 4, 4]
        tomo_probs = list(map(self.get_tomo_from_density_matrix, projectors))
        tomo_probs = np.array(tomo_probs).flatten()
        draw = np.random.multinomial(1, tomo_probs)
        arg_max_rank = np.argmax(draw)
        measurements = np.zeros(6 ** self._qs).astype(np.float32)
        measurements[2**self._qs * proj_int + arg_max_rank] = 1.
        if proj_rank_list:
            self.proj_used_rank_list.append(2**self._qs * proj_int + arg_max_rank)
        return measurements

    def measurement_array(self, dm=None):
        dm = np.squeeze(dm)
        proj_rank = np.random.randint(0, 3**self._qs, self.n_shots)
        measurement_list = list(map(lambda x: self.measure_specific_basis(x, dm), proj_rank))
        measurement_array = np.array(measurement_list).reshape(-1, 6 ** self._qs)
        self.count += 1
        if self.count % 1000 == 0:
            print(f'dm_finished: {self.count}')
        return measurement_array

    def tomography_data(self, density_matrix, norm=True, save_file=False, filename='pickle.pickle'):
        self.count = 1
        measurements = list(map(self.measurement_array, density_matrix))
        measurements = np.array(measurements).reshape(-1, self.n_shots, 6 ** self._qs)
        if norm:
            tomo_array = np.sum(measurements, axis=1) / self.n_shots
        else:
            tomo_array = np.sum(measurements, axis=1)
        tau_list = list(map(self.get_tau_cholesky, density_matrix))
        tau_array = np.array(tau_list).reshape(-1, 2 ** self._qs * 2 ** self._qs)

        if save_file:
            if not os.path.exists('./data/data_measurements'):
                os.mkdir('./data/data_measurements')
                print('./data/data_measurements folder has been created and ')
            with open(f'./data/measurements/{filename}', 'wb') as f:
                print(f'tomography data has been saved into ./data/data_measurements/{filename}')
                pkl.dump([tomo_array, tau_array, density_matrix, self.proj_used_rank_list], f, -1)

        # return measurements, self.proj_used_rank_list
        return tomo_array, tau_array