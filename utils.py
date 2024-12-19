import numpy as np
from scipy.sparse import csr_matrix

spin_states = { 
    '1': 3,
    '1/2': 2,
}

spin_operators = {
    '1': {
        'Sz' : csr_matrix( np.array([[1,0,0],[0,0,0],[0,0,-1]])),
        'Sz2' : csr_matrix(np.array([[1,0,0],[0,0,0],[0,0,1]])),
        'Sx' : csr_matrix(1/np.sqrt(2)*np.array([[0,1,0],[1,0,1],[0,1,0]])),
        'Sy' : csr_matrix(1j/np.sqrt(2)*np.array([[0,-1,0],[1,0,-1],[0,1,0]])),
    },

    '1/2': {
        'Sz' : csr_matrix(np.array([[1,0],[0,-1]])/2),
        'Sx' : csr_matrix( np.array([[0,1],[1,0]])/2),
        'Sy' : csr_matrix(np.array([[0,-1j],[1j,0]])/2),
    }
}