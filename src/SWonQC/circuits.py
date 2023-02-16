import sys
import numpy as np
from qiskit import QuantumCircuit

def CU_trotterized(operator):
    """
    Implement e^operator on a quantum computer.
    Compared to Cirq, the identities are written in the Pauli operators,
    and one has to be careful about the ordering of the Pauli operator !
    Indeed, operator[i].to_pauli_op().primitive[nqubits-j-1] actually correspond to qubit j !
    Arguments:
     - operator: the qubit operator to be exponentiated and trotterized.
    Returns the circuit.
    """
    rot_dic = { 'X' : lambda qubit, sign : circuit.h(qubit),
                'Y' : lambda qubit, sign : circuit.rx(sign*np.pi/2., qubit)}
    nterms=len(operator)
    nqubits=operator.num_qubits
    circuit = QuantumCircuit(nqubits)
    for i in range(nterms):
        cnot_qubit_pairs = []
        # Begining of the circuit fragment: we detect if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        for j in range(nqubits):
            if str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'I':
                continue
            elif str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'Z':
                cnot_qubit_pairs.append(j)
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[nqubits-j-1])](j,1)
                cnot_qubit_pairs.append(j)
        # Middle of the circuit fragment
        # First, We extend the chain of Z operators thoughout the required
        # qubits to match the length of the Pauli string
        if len(cnot_qubit_pairs) > 1 and nqubits > 1:
            for j in range(len(cnot_qubit_pairs)-1):
                circuit.cx(cnot_qubit_pairs[j],cnot_qubit_pairs[j+1])
        # Second, we apply a Z rotation gate on the qubit with the last non-identity gate of the Pauli string.
        if len(cnot_qubit_pairs) > 0: # the length is zero only if there were only identities.
            circuit.rz((+2.0*operator[i].to_pauli_op().coeff.imag),cnot_qubit_pairs[-1])
            # factor 2 because Rz(lambda) ---> e^[i lambda/2].
        # Then, we extend again the chain of Z operators thoughtout the required
        # qubits to match the length of the Pauli string
        if len(cnot_qubit_pairs) > 1 and nqubits > 1:
            for j in range(len(cnot_qubit_pairs)-2,-1,-1):
                circuit.cx(cnot_qubit_pairs[j],cnot_qubit_pairs[j+1])
        # Finally,  we detect again if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        for j in range(nqubits-1,-1,-1):
            if str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'I' or str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'Z':
                continue
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[nqubits-j-1])](j,-1)

    return circuit

def initial_circ(n_sites,state='Neel',phase=False,varphi=np.pi):
    '''
    Create the circuit associated to the AntiFerroMagnetic Neel state, or Ionic state.
    '''
    initial_circuit = QuantumCircuit(2*n_sites)

    if state == "Neel": # cos(varphi/2) |100110011001...>  - sin(varphi/2) |0110011001100110...>) (Check that S^2 = 0)
       initial_circuit.ry(varphi,0)
       initial_circuit.x(1)
       initial_circuit.x(3)
       initial_circuit.cx(0,1)
       initial_circuit.cx(1,2)
       initial_circuit.cx(2,3)
       if phase: initial_circuit.z(3)
       for i in range(1,n_sites//2):
         initial_circuit.cx(3,4*i)
         initial_circuit.cx(3,4*i+3)
         initial_circuit.cx(2,4*i+1)
         initial_circuit.cx(2,4*i+2)
    elif state == "Ionic": # cos(varphi/2) |110011001100...> + sin(varphi/2) |001100110011...>
       initial_circuit.ry(varphi,0)
       initial_circuit.x(2)
       initial_circuit.cx(0,1)
       initial_circuit.cx(1,2)
       initial_circuit.cx(2,3)
       if phase: initial_circuit.z(3)
       for i in range(1,n_sites//2):
         initial_circuit.cx(0,4*i)
         initial_circuit.cx(0,4*i+1)
         initial_circuit.cx(2,4*i+2)
         initial_circuit.cx(2,4*i+3)
    else:
       sys.exit('Wrong initial state, not implemented.')

    return initial_circuit
