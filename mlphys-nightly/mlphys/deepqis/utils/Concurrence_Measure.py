import numpy as np
import qiskit.quantum_info as qi


def concurrence_single(dm):
    # dm = qi.DensityMatrix(dm)
    con = qi.concurrence(dm)
    return con


def concurrence(dm_tensor):
    con_list = list(map(concurrence_single, dm_tensor))
    con_tensor = np.array(con_list)
    return con_tensor
