import utils;
import numpy as np
from scipy.sparse import csr_array, kron, eye, linalg

class Hamiltonian:
    def __init__(self, n, spin='1'):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system
        spin: str
            Spin value of the system
        """

        self._n = n
        self._spin = spin
        self._matrix =csr_array((utils.spin_states[spin]**self._n, 
            utils.spin_states[spin]**self._n), dtype=np.complex64)
        self._kroned_identities = [ 
                eye(utils.spin_states[spin]**i) for i in range(0,n-1)
            ]
        self._gstate = None # ground state
        self._gstate_wf = None # ground state wavefunction

    @property
    def n(self) -> int:
        return self._n
    @property
    def spin(self) -> str:
        return self._spin
    @property
    def matrix_dim(self) -> int:
        return self._matrix.shape[0]
    @property
    def gstate(self) -> np.complex128:
        if self._gstate is None:
            self._gstate, self._gstate_wf = linalg.eigs(self._matrix, k=1, which='LM')
        return self._gstate
    
    @property
    def gstate_wf(self):
        if self._gstate_wf is None:
            self._gstate
        return self._gstate_wf

    def kroned_identity(self, index) -> np.array:
        
        return self._kroned_identities[index]

    def _build_term(self, i, operator, coeff):
        return kron(
        kron(self.kroned_identity(i), operator),
        kron(operator, self.kroned_identity(self._n - 2 - i))
    ) * coeff

    def _cyclical_term(self, operator, coeff):
        return kron(
            operator, kron(
            self.kroned_identity(self._n - 2),
            operator)
        ) * coeff

    def __str__(self):
        return f"Hamiltonian of {self.n} spins of value {self.spin}"

class BondAlternatingXXZ(Hamiltonian):
    def __init__(self, n, Delta, delta, spin='1'):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system
        Delta : float
            Anisotropy parameter
        delta : float
            Alternation parameter
        spin: str
            Spin value of the system.
        """
        super().__init__(n, spin)
        self._Delta = Delta 
        self._delta = delta

        # Build the Hamiltonian matrix for the non-cyclic terms
        for l in range(n-1):
            coeff = 1 - delta * (-1)**(l+1)
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sx'], coeff)   # S_l^x S_{l+1}^x term
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sy'], coeff)   # S_l^y S_{l+1}^y term
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sz'], Delta * coeff)   # S_l^z S_{l+1}^z term

        # Cyclical terms
        coeff = 1 - delta * (-1)**n
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sx'], coeff)  # S_N^x S_1^x term
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sy'], coeff)  # S_N^y S_1^y term  
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sz'], Delta * coeff) # S_N^z S_1^z term

        self._matrix = csr_array(self._matrix)
    
    @property
    def Delta(self):
        return self._Delta
    @property
    def delta(self):
        return self._Delta
    
    def __str__(self):
        return super().__str__()  +  \
        f"""\nBond-alternating XXZ chain\n\nHamiltonian properties:
        δ = {self.delta} 
        Δ = {self.Delta}.\n"""
        

class XXZUniaxialSingleIonAnisotropy(Hamiltonian):
    def __init__(self, n, Jz, D, spin='1', J=1):
        """
        Parameters
        ----------
        n : int
            Number of spins in the system
        J : float
            Spin coupling parameter in the xy-axis
        Jz : float
            Spin coupling parameter in the z-axis
        D : float
            Uniaxial single ion anisotropy parameter
        spin: str
            Spin value of the system.
        """
        super().__init__(n, spin)
        self._Jz = Jz 
        self._D = D

        # Build the Hamiltonian matrix for the non-cyclic terms
        for l in range(n-1):
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sx'], J)   # S_l^x S_{l+1}^x term
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sy'], J)   # S_l^y S_{l+1}^y term
            self._matrix += self._build_term(l, utils.spin_operators[spin]['Sz'], Jz)   # S_l^z S_{l+1}^z term
            self._matrix += self._build_anisotropy(l, utils.spin_operators[spin]['Sz2'], D)   # S_l^z2 term
        # Cyclical terms
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sx'], J)  # S_N^x S_1^x term
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sy'], J)  # S_N^y S_1^y term  
        self._matrix += self._cyclical_term(utils.spin_operators[spin]['Sz'], Jz) # S_N^z S_1^z term
        self._matrix += self._build_anisotropy(n-1, utils.spin_operators[spin]['Sz2'], D)   # S_l^z2 term

        self._matrix = csr_array(self._matrix)
    
    def _build_anisotropy(self, i, operator, coeff):
        return kron(
        kron(self.kroned_identity(i), operator),
        self.kroned_identity(self._n - 1 - i)
    ) * coeff
    
    @property
    def Delta(self):
        return self._Delta
    @property
    def delta(self):
        return self._Delta
    
    def __str__(self):
        return super().__str__()  +  \
        f"""XXZ chains with uniaxial single-ion-type anisotropy,\n\nHamiltonian properties:
        Jz = {self.Jz} 
        D = {self.D}.\n"""


x = BondAlternatingXXZ(8, 1, 1)

eigvalues, eigvec =  linalg.eigs(x._matrix, k = 4)

print(eigvalues)
del x