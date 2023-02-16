import sys,os
import numpy as np
import math
import scipy

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.algorithms import NumPyEigensolver
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeBogota, FakeMontreal, FakeManila
import qiskit.providers.aer.noise as noise

import SWonQC

SWonQC_directory = os.getenv('SWonQC_DIR')
results_folder = SWonQC_directory + '/examples/results/'

# For the moment, only the ABAB 1D Hubbard model is implemented, with potential -deltav/2, +deltav/2, -deltav/2, +deltav/2, ..., with deltav > 0

###############################################
#              INITIALIZATION                 #
###############################################

# system
n_sites = 2
t = 1.
deltav_list = [0.0]
number_of_points = 2

# initial state
initial_state = ["Neel","Ionic"][0]
varphi = ["equi","pur"][0]
phase = True

###############################################
#                  CIRCUIT                    #
###############################################
backend = QasmSimulator(method='statevector')
n_qubits = 2*n_sites

if varphi == "equi": 
   varphi_value = np.pi/2.
   initial_circuit = SWonQC.initial_circ(n_sites,state=initial_state,phase=phase,varphi=varphi_value)

elif varphi == "pur": 
   varphi_value = np.pi
   initial_circuit = SWonQC.initial_circ(n_sites,state=initial_state,phase=phase,varphi=varphi_value)

for deltav in deltav_list:
    output_file = results_folder + "1DHubbard_noiseless_{}sites_t{}_deltav{}_iterative_{}_varphi{}_phase{}.dat".format(n_sites,t,deltav,initial_state,varphi,phase)

    with open(output_file,'w') as f: 
         f.write('{:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s}\n'.format("U","E0_exact","E1_exact","E2_exact","energy","s2","iteration"))

    for U_Ut_npoints in range(1,number_of_points+1):

        U_Ut = U_Ut_npoints/float(number_of_points+1)
        if n_sites == 2: 
           U = 2*t*U_Ut/(1 - U_Ut)
        else: 
           U = 4*t*U_Ut/(1 - U_Ut)

        ###############################################
        #               HAMILTONIAN                   #
        ###############################################

        hspace = []
        for i in range(2**n_qubits):
         if SWonQC.count_ones_bitstring(i) == n_sites:
           hspace += [i]

        Hubbard_operator = SWonQC.Hubbard_1D_operator(n_sites,U,t,deltav)
        jw_mapper = JordanWignerMapper()
        jw_converter = QubitConverter(jw_mapper)
        Hubbard_PauliSum = jw_converter.convert(Hubbard_operator)
        
        solver  = NumPyEigensolver(k = 3)
        result  = solver.compute_eigenvalues(Hubbard_PauliSum)
        eigvals = result.eigenvalues
        
        # Operator(Hubbard_PauliSum).data and Hubbard_operator.to_matrix().A give a different ordered Hmatrix due to different endian ordering.
        #Hubbard_matrix = Operator(Hubbard_PauliSum).data
        # I don't manage to make it work with Hubbard_PauliSum in every case... So I switch to Hfermion.to_matrix, which always work (for trotter and untrotter) with reverse_bits().
        Hubbard_matrix = Hubbard_operator.to_matrix().A
        eigval_exact, eigvec_exact = np.linalg.eigh(Hubbard_matrix)
        Hubbard_matrix_hspace=np.array([[Hubbard_matrix[i,j] for i in hspace] for j in hspace])
        eigval_hspace,eigvec_hspace = np.linalg.eigh(Hubbard_matrix_hspace)

        # keep only singlet states:
        s2_op = SWonQC.s2_operator(n_qubits)
        s2_PauliSum = jw_converter.convert(s2_op)
        #s2_matrix = Operator(s2_PauliSum).data
        s2_matrix = s2_op.to_matrix().A # not same endian-ordering as Hmatrix and SW_matrix...
        s2_matrix_hspace=np.array([[s2_matrix[i,j] for i in hspace] for j in hspace])
        #print("Test S2:",eigvec_hspace[:,0].T @ s2_matrix_hspace @ eigvec_hspace[:,0])
        eigval = []
        for i in range(len(hspace)):
          s2 = int((eigvec_hspace[:,i].T @ s2_matrix_hspace @ eigvec_hspace[:,i]).real)
          if s2 == 0:
            eigval.append(eigval_hspace[i])

        ###############################################
        #               SW operator                   #
        ###############################################

        Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2 = SWonQC.J_terms_initialization(n_sites,t)
        #lambda_0,lambda_1,lambda_2,lambda_3 = SWonQC.lambda_terms_matrixsolver(n_sites,U,deltav,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2)
        lambda_0,lambda_1,lambda_2,lambda_3 = SWonQC.lambda_terms(n_sites,U,deltav,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2)
        SW_operator = SWonQC.SW_operator(n_sites,lambda_0,lambda_1,lambda_2,lambda_3)
        SW_PauliSum = jw_converter.convert(SW_operator)

        # Initialize the iterative process by applying the standard SW transformation on the original problem
        # Iteration 0 is already done before.
        # We now need to iterate to get the SW_op at iteration (n)
        SW_PauliSum_list = [SW_PauliSum]
        iteration = 0
        while np.linalg.norm(Jcpl_0 + Jcpl_1 + Jcpl_2 + Jcpl_3) > 1e-7:
            iteration += 1
            Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2 = SWonQC.J_terms(n_sites,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2,lambda_0,lambda_1,lambda_2,lambda_3)
            lambda_0,lambda_1,lambda_2,lambda_3 = SWonQC.lambda_terms(n_sites,U,deltav,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2)

            SW_operator_iteration = SWonQC.SW_operator(n_sites,lambda_0,lambda_1,lambda_2,lambda_3)
            SW_PauliSum = jw_converter.convert(SW_operator_iteration)
            SW_PauliSum_list.append(SW_PauliSum)

        ###############################################
        #      classical noiseless simulations        #
        ###############################################

        energy = SWonQC.evaluate_SWiterative_statevector(initial_circuit,SW_PauliSum_list,Hubbard_matrix,backend)
        s2 = SWonQC.evaluate_SWiterative_statevector(initial_circuit,SW_PauliSum_list,s2_matrix,backend)

        print("%"*50+'\n')
        print("Delta v = {}, U = {}\n".format(deltav,U))
        print("Energies:")
        print(" Exact: {}".format(eigval[0]))
        print(" Trotterized: {}".format(energy))
        print("s2: {}".format(s2))
        print("%"*50)

        with open(output_file,'a') as f:
             f.write('{:<20.4f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20d}\n'.format(U,eigval[0],eigval[1],eigval[2],energy,s2,iteration))
