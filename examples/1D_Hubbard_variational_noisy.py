import sys,os
import numpy as np
import math
import scipy

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.algorithms.optimizers import SPSA
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

# classical optimization
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][3]

# noise
nshots = 1024
noisy_simulation = ["Model","FakeManila"][0]

if noisy_simulation == "FakeManila":
  device_backend   = FakeManila()
  device           = QasmSimulator.from_backend(device_backend)
  noise_model      = NoiseModel.from_backend(device)

if noisy_simulation == "Model":
  # Error probabilities
  prob_1 = 0.0001  # 1-qubit gate
  prob_2 = 0.001   # 2-qubit gate
  # Depolarizing quantum errors
  error_1 = noise.depolarizing_error(prob_1, 1)
  error_2 = noise.depolarizing_error(prob_2, 2)
  # Add errors to noise model
  noise_model = noise.NoiseModel()
  noise_model.add_all_qubit_quantum_error(error_1, ['u1','u2','u3','ry','rz','sx'])
  noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

###############################################
#                  CIRCUIT                    #
###############################################
backend = QasmSimulator(method='statevector', noise_model=noise_model)
n_qubits = 2*n_sites

if varphi == "equi": 
   varphi_value = np.pi/2.
   initial_circuit = SWonQC.initial_circ(n_sites,state=initial_state,phase=phase,varphi=varphi_value)

elif varphi == "pur": 
   varphi_value = np.pi
   initial_circuit = SWonQC.initial_circ(n_sites,state=initial_state,phase=phase,varphi=varphi_value)

for deltav in deltav_list:

    if noisy_simulation == "Model": 
       output_file = results_folder + "1DHubbard_noisy_{}sites_t{}_deltav{}_variational_opt{}_{}_varphi{}_phase{}_nshots{}_p1g{}_p2g{}.dat".format(n_sites,t,deltav,opt_method,initial_state,varphi,phase,nshots,prob_1,prob_2)
    elif noisy_simulation == "FakeManila": 
       output_file = results_folder + "1DHubbard_noisy_{}sites_t{}_deltav{}_variational_opt{}_{}_varphi{}_phase{}_nshots{}_FakeManila.dat".format(n_sites,t,deltav,opt_method,initial_state,varphi,phase,nshots)

    with open(output_file,'w') as f: 
         f.write('{:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s} {:20s}\n'.format("U","E0_exact","E1_exact","E2_exact","energy","energy_Neel","energy_IonicPur","energy_IonicEqui","theta","s2"))

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

        ###############################################
        #          Classical Optimization             #
        ###############################################

        theta_homogeneous = (U/4*t)*np.arctan(4*t/U)
        if opt_method != "SPSA": 
          result = scipy.optimize.minimize(SWonQC.evaluate,
                                                     x0=[.4],
                                                     args=(initial_circuit,
                                                           SW_PauliSum,
                                                           Hubbard_PauliSum,
                                                           backend,
                                                           nshots),
                                                     method=opt_method,
                                                     #bounds=[(0.,1.)],
                                                     options={'ftol':1e-14,'gtol': 1e-09})
          theta_opt = result.x[0]
          energy = result.fun
        else:
          #spsa = SPSA(maxiter=500,blocking=True,allowed_increase=0.1,last_avg=25,resamplings=2)
          spsa = SPSA(maxiter=1000,last_avg=25)
          cost_function = SPSA.wrap_function(SWonQC.evaluate,
                                             (initial_circuit,
                                              SW_PauliSum,
                                              Hubbard_PauliSum,
                                              backend,
                                              nshots))
          result = spsa.minimize(cost_function, x0=[0.4])
          theta_opt = result.x[0]
          energy = result.fun # if last_avg is not 1, it returns the callable function with the last_avg param_values as input ! This seems generally good and better than taking the mean of the last_avg function calls.
          stddev = spsa.estimate_stddev(cost_function, initial_point=[theta_opt])

        ###############################################
        #      100 classical noisy simulations        #
        ###############################################
        # noisy: 100 simulations with the theta_opt input, and average to get the estimation of the energy.
        s2 = 0
        energy_Neel = 0
        energy_IonicPur = 0
        energy_IonicEqui = 0
        SW_PauliSum_theta = SW_PauliSum.mul(theta_opt)
        Neel_circuit = SWonQC.initial_circ(n_sites,state='Neel',phase='True',varphi=np.pi/2.)
        IonicPur_circuit = SWonQC.initial_circ(n_sites,state='Ionic',phase='False',varphi=np.pi)
        IonicEqui_circuit = SWonQC.initial_circ(n_sites,state='Ionic',phase='False',varphi=np.pi/2.)

        for i in range(100):
            total_circuit = initial_circuit.compose(SWonQC.CU(SW_PauliSum_theta))
            s2 += SWonQC.sampled_expectation_value(total_circuit,s2_PauliSum,backend,nshots=nshots)
            total_circuit = Neel_circuit.compose(SWonQC.CU(SW_PauliSum_theta))
            energy_Neel += SWonQC.sampled_expectation_value(total_circuit,Hubbard_PauliSum,backend,nshots=nshots)
            total_circuit = IonicPur_circuit.compose(SWonQC.CU(SW_PauliSum_theta))
            energy_IonicPur += SWonQC.sampled_expectation_value(total_circuit,Hubbard_PauliSum,backend,nshots=nshots)
            total_circuit = IonicEqui_circuit.compose(SWonQC.CU(SW_PauliSum_theta))
            energy_IonicEqui += SWonQC.sampled_expectation_value(total_circuit,Hubbard_PauliSum,backend,nshots=nshots)

        s2 = s2 / 100.
        energy_Neel = energy_Neel/100.
        energy_IonicPur = energy_IonicPur/100.
        energy_IonicEqui = energy_IonicEqui/100.
           
        print("%"*50+'\n')
        print("Delta v = {}, U = {}\n".format(deltav,U))
        print("Energies:")
        print("  Exact: {}".format(eigval[0]))
        print("  Trotterized: {}".format(energy))
        print("  Trotterized Neel: {}".format(energy_Neel))
        print("  Trotterized IonicPur: {}".format(energy_IonicPur))
        print("  Trotterized IonicEqui: {}".format(energy_IonicEqui))
        print("Optimal theta value:")
        print("  Exact (homogeneous): {}".format(theta_homogeneous))
        print("  Trotterized: {}".format(theta_opt))
        print("s2: {}".format(s2))
        print("%"*50)

        with open(output_file,'a') as f:
             f.write('{:<20.4f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f} {:<20.12f}\n'.format(U,eigval[0],eigval[1],eigval[2],energy,energy_Neel,energy_IonicPur,energy_IonicEqui,theta_opt,s2))
