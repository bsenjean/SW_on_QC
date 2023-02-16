import sys
import numpy as np
from qiskit_nature.operators.second_quantization import FermionicOp

def Hubbard_1D_operator(n_sites,U,t,deltav):
    '''
    1D Hubbard operator in qiskit.   
    '''

    n_qubits = 2*n_sites
    Hloc_op = 0
    V_op = 0

    for i in range(n_sites//2):
      Hloc_op += FermionicOp(("N_{}".format(4*i),-deltav/2.),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{}".format(4*i+1),-deltav/2.),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{}".format(4*i+2),+deltav/2.),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{}".format(4*i+3),+deltav/2.),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{} N_{}".format(2*i,2*i+1),U),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{} N_{}".format(n_sites+2*i,n_sites+2*i+1),U),register_length=n_qubits)
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
            V_op += FermionicOp(("+_{} -_{}".format(2*i,2*j),-t),register_length=n_qubits)
            V_op += FermionicOp(("+_{} -_{}".format(2*i+1,2*j+1),-t),register_length=n_qubits)

    return Hloc_op + V_op

def sz_operator(n_qubits):
    s_z = 0

    for i in range(n_qubits//2):
      s_z += FermionicOp(("N_{}".format(2*i),0.5),register_length=n_qubits)
      s_z -= FermionicOp(("N_{}".format(2*i+1),0.5),register_length=n_qubits)

    return s_z
    
def s2_operator(n_qubits):
    ''' 
    S2 = S- S+ + Sz(Sz+1)
    I use the usual sorting as in OpenFermion, i.e. 1up 1down, 2up 2down, etc...
    '''
    s2_op = 0
    s_moins = 0
    s_plus = 0
    s_z = sz_operator(n_qubits)
    
    for i in range(n_qubits//2):
      s_moins += FermionicOp(("+_{} -_{}".format(2*i+1,2*i),1),register_length=n_qubits)
      s_plus += FermionicOp(("+_{} -_{}".format(2*i,2*i+1),1),register_length=n_qubits)
      
    s2_op = s_moins @ s_plus + s_z @ s_z + s_z
    return s2_op

def SW_operator(n_sites,lambda_0,lambda_1,lambda_2,lambda_3) -> FermionicOp:
    '''
    Returns the SW operator for the inhomogeneous 1D Hubbard chain.
    I think it only works with periodic condition right now ?
    '''

    n_qubits = 2*n_sites
    SW_op = 0

    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          Pij0_up   = lambda_0[i,j]*(FermionicOp(("I",1),register_length=n_qubits)\
                              + FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits)\
                              - FermionicOp("N_{}".format(2*i),register_length=n_qubits)\
                              - FermionicOp("N_{}".format(2*j),register_length=n_qubits))
          Pij1_up   = lambda_1[i,j]*(FermionicOp("N_{}".format(2*i),register_length=n_qubits)\
                              - FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
          Pij2_up   = lambda_2[i,j]*(FermionicOp("N_{}".format(2*j),register_length=n_qubits)\
                              - FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
          Pij3_up   = lambda_3[i,j]*(FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
          Pij0_down = lambda_0[i,j]*(FermionicOp(("I",1),register_length=n_qubits)\
                              + FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits)\
                              - FermionicOp("N_{}".format(2*i+1),register_length=n_qubits)\
                              - FermionicOp("N_{}".format(2*j+1),register_length=n_qubits))
          Pij1_down = lambda_1[i,j]*(FermionicOp("N_{}".format(2*i+1),register_length=n_qubits)\
                              - FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))
          Pij2_down = lambda_2[i,j]*(FermionicOp("N_{}".format(2*j+1),register_length=n_qubits)\
                              - FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))
          Pij3_down = lambda_3[i,j]*(FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))

          cicj_up   = FermionicOp("+_{} -_{}".format(2*i  ,2*j  ),register_length=n_qubits)
          cicj_down = FermionicOp("+_{} -_{}".format(2*i+1,2*j+1),register_length=n_qubits)
          cjci_up   = FermionicOp("+_{} -_{}".format(2*j  ,2*i  ),register_length=n_qubits)
          cjci_down = FermionicOp("+_{} -_{}".format(2*j+1,2*i+1),register_length=n_qubits)

          Pij_up = Pij0_up + Pij1_up + Pij2_up + Pij3_up
          Pij_down = Pij0_down + Pij1_down + Pij2_down + Pij3_down

          SW_op += 0.25*(Pij_up@(cjci_down - cicj_down) + Pij_down@(cjci_up - cicj_up))

    if n_sites == 2: SW_op = 2*SW_op

    return SW_op

def lambda_terms_0(n_sites,t,U,deltav):
    '''
    Compute lambda terms at Oth order.
    '''

    potential = np.zeros(n_sites)
    for i in range(n_sites//2):
      potential[2*i]   = -deltav/2.
      potential[2*i+1] = +deltav/2.

    lambda_0 = np.zeros([n_sites,n_sites])
    lambda_1 = np.zeros([n_sites,n_sites])
    lambda_2 = np.zeros([n_sites,n_sites])
    lambda_3 = np.zeros([n_sites,n_sites])
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          if potential[i] == potential[j]:
            lambda_1[i,j] = -2*t/U
            lambda_2[i,j] = 2*t/U
          else:
            lambda_0[i,j] = 2*t/(potential[j] - potential[i])
            if (potential[j] - potential[i])!=U:
               lambda_1[i,j] = 2*t/(potential[j] - potential[i] - U)
            if (potential[j] - potential[i])!=-U:
               lambda_2[i,j] = 2*t/(potential[j] - potential[i] + U)
            lambda_3[i,j] = 2*t/(potential[j] - potential[i])

    if n_sites == 2:
      lambda_0 = lambda_0/2.
      lambda_1 = lambda_1/2.
      lambda_2 = lambda_2/2.
      lambda_3 = lambda_3/2.

    return lambda_0,lambda_1,lambda_2,lambda_3

def lambda_terms(n_sites,U,deltav,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2):
    '''
    Compute lambda terms at iteration n from the J of iteration n-1.
    '''

    potential = np.zeros(n_sites)
    for i in range(n_sites//2):
      potential[2*i]   = -deltav/2.
      potential[2*i+1] = +deltav/2.

    lambda_0 = np.zeros([n_sites,n_sites])
    lambda_1 = np.zeros([n_sites,n_sites])
    lambda_2 = np.zeros([n_sites,n_sites])
    lambda_3 = np.zeros([n_sites,n_sites])
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          B1 = (potential[i] - potential[j] - U) + 3*Jdia_1[i,j] - Jdia_2[i,j] + 2*Jex_1[i,j]
          B2 = (potential[i] - potential[j] + U) + 3*Jdia_2[i,j] - Jdia_1[i,j] - 2*Jex_2[i,j]
          if (potential[i] - potential[j])/2 == -2*Jdia_0[i,j]:
            lambda_0[i,j] = 0
          else:
            lambda_0[i,j] = Jcpl_0[i,j]/(potential[i] - potential[j] + 2*Jdia_0[i,j])
          if 4*Jde_1[i,j]*Jde_2[i,j] == B1*B2:
            lambda_1[i,j] = 0
            lambda_2[i,j] = 0
          else:
            lambda_1[i,j] = (-2*Jcpl_2[i,j]*Jde_1[i,j] + Jcpl_1[i,j]*B2)/(4*Jde_1[i,j]*Jde_2[i,j] + B1*B2)
            lambda_2[i,j] = (2*Jcpl_1[i,j]*Jde_2[i,j] + Jcpl_2[i,j]*B1)/(4*Jde_1[i,j]*Jde_2[i,j] + B1*B2)
          if (potential[i] - potential[j])/2 == -2*Jdia_3[i,j]:
            lambda_3[i,j] = 0
          else:
            lambda_3[i,j] = Jcpl_3[i,j]/(potential[i] - potential[j] + 2*Jdia_3[i,j])

    return lambda_0,lambda_1,lambda_2,lambda_3

def lambda_terms_matrixsolver(n_sites,U,deltav,Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2):
    '''
    Another way of computing lambda_SW such that [S, H0] = SH0 = V, --> lambda = V @ SH0^(-1)
    '''

    deltav = deltav + 1.e-15 # Avoid numerical instability for exact calculation
    potential = np.zeros(n_sites)
    for i in range(n_sites//2):
      potential[2*i]   = -deltav/2.
      potential[2*i+1] = +deltav/2.

    lambda_0 = np.zeros([n_sites,n_sites])
    lambda_1 = np.zeros([n_sites,n_sites])
    lambda_2 = np.zeros([n_sites,n_sites])
    lambda_3 = np.zeros([n_sites,n_sites])
    SH0 = np.zeros([4,4])
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          #SH0[0,0] = - deltav + 2*Jdia_0[i,j]
          SH0[0,0] = potential[i] - potential[j] + 2*Jdia_0[i,j]
          #SH0[1,1] = - U - deltav + 3*Jdia_1[i,j] - Jdia_2[i,j] + 2*Jex_1[i,j]
          SH0[1,1] = - U + potential[i] - potential[j] + 3*Jdia_1[i,j] - Jdia_2[i,j] + 2*Jex_1[i,j]
          SH0[2,1] = 2*Jde_1[i,j] # or Jde_2?
          #SH0[2,2] = U - deltav + 3*Jdia_2[i,j] - Jdia_1[i,j] - 2*Jex_2[i,j]
          SH0[2,2] = U + potential[i] - potential[j] + 3*Jdia_2[i,j] - Jdia_1[i,j] - 2*Jex_2[i,j]
          SH0[1,2] = -2*Jde_2[i,j] # or Jde_1?
          #SH0[3,3] = - deltav + 2*Jdia_3[i,j]
          SH0[3,3] = potential[i] - potential[j] + 2*Jdia_3[i,j]
          lbd_sw   = np.array([Jcpl_0[i,j],Jcpl_1[i,j],Jcpl_2[i,j],Jcpl_3[i,j]]) @ np.linalg.inv(SH0)
          lambda_0[i,j] = lbd_sw[0]
          lambda_1[i,j] = lbd_sw[1]
          lambda_2[i,j] = lbd_sw[2]
          lambda_3[i,j] = lbd_sw[3]

    return lambda_0,lambda_1,lambda_2,lambda_3

def J_terms_initialization(n_sites,t):
    Jcpl_0 = np.zeros([n_sites,n_sites])
    Jcpl_1 = np.zeros([n_sites,n_sites])
    Jcpl_2 = np.zeros([n_sites,n_sites])
    Jcpl_3 = np.zeros([n_sites,n_sites])
    Jdia_0 = np.zeros([n_sites,n_sites])
    Jdia_1 = np.zeros([n_sites,n_sites])
    Jdia_2 = np.zeros([n_sites,n_sites])
    Jdia_3 = np.zeros([n_sites,n_sites])
    Jex_1  = np.zeros([n_sites,n_sites])
    Jex_2  = np.zeros([n_sites,n_sites])
    Jde_1  = np.zeros([n_sites,n_sites])
    Jde_2  = np.zeros([n_sites,n_sites])
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          Jcpl_0[i,j] = -t
          Jcpl_1[i,j] = -t
          Jcpl_2[i,j] = -t
          Jcpl_3[i,j] = -t

    return Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2

def J_terms(n_sites,Jdia_previous_0,Jdia_previous_1,Jdia_previous_2,Jdia_previous_3,Jcpl_previous_0,Jcpl_previous_1,Jcpl_previous_2,Jcpl_previous_3,Jde_previous_1,Jde_previous_2,Jex_previous_1,Jex_previous_2,lambda_0,lambda_1,lambda_2,lambda_3):
    '''
    Compute the J terms of iteration n from the lambda of iteration n (determined from the J of iteration n-1).
    '''

    Jdia_0 = np.zeros([n_sites,n_sites])
    Jdia_1 = np.zeros([n_sites,n_sites])
    Jdia_2 = np.zeros([n_sites,n_sites])
    Jdia_3 = np.zeros([n_sites,n_sites])
    Jcpl_0 = np.zeros([n_sites,n_sites])
    Jcpl_1 = np.zeros([n_sites,n_sites])
    Jcpl_2 = np.zeros([n_sites,n_sites])
    Jcpl_3 = np.zeros([n_sites,n_sites])
    Jde_1  = np.zeros([n_sites,n_sites])
    Jde_2  = np.zeros([n_sites,n_sites])
    Jex_1  = np.zeros([n_sites,n_sites])
    Jex_2  = np.zeros([n_sites,n_sites])
    for i in range(n_sites):
      for j in range(n_sites):
        if i==j+1 or i==j-1 or (i==0 and j==n_sites-1) or (i==n_sites-1 and j==0): # periodic boundary conditions
          W1 = lambda_2[i,j]*Jcpl_previous_1[i,j] + lambda_1[i,j]*Jcpl_previous_2[i,j]
          W2 = lambda_1[i,j]*Jcpl_previous_1[i,j] - lambda_2[i,j]*Jcpl_previous_2[i,j]
          alpha = np.sqrt((lambda_1[i,j]**2 + lambda_2[i,j]**2)/2.)
          Jdia_0[i,j] = Jcpl_previous_0[i,j]*(np.sin(2*lambda_0[i,j]) - lambda_0[i,j]*np.sinc(lambda_0[i,j]/np.pi)**2)
          if lambda_1[i,j] != 0 and lambda_1[i,j] != lambda_2[i,j]:
            Jdia_1[i,j] = (1./(1 + lambda_2[i,j]**2/lambda_1[i,j]**2))*\
                          (2*lambda_2[i,j]*W1*np.sinc(2*alpha/np.pi)/lambda_1[i,j] + 2*W2*np.sinc(4*alpha/np.pi) \
                           - lambda_2[i,j]*W1*np.sinc(alpha/np.pi)**2/lambda_1[i,j] - W2*np.sinc(2*alpha/np.pi)**2)
          if lambda_2[i,j] != 0 and lambda_1[i,j] != lambda_2[i,j]:
            Jdia_2[i,j] = (1./(1 + lambda_1[i,j]**2/lambda_2[i,j]**2))*\
                          (2*lambda_1[i,j]*W1*np.sinc(2*alpha/np.pi)/lambda_2[i,j] - 2*W2*np.sinc(4*alpha/np.pi) \
                           - lambda_1[i,j]*W1*np.sinc(alpha/np.pi)**2/lambda_2[i,j] + W2*np.sinc(2*alpha/np.pi)**2)
          Jdia_3[i,j] = Jcpl_previous_3[i,j]*(np.sin(2*lambda_3[i,j]) - lambda_3[i,j]*np.sinc(lambda_3[i,j]/np.pi)**2)
          Jcpl_0[i,j] = Jcpl_previous_0[i,j]*(np.cos(2*lambda_0[i,j]) - np.sinc(2*lambda_0[i,j]/np.pi))
          if lambda_1[i,j] != 0 and lambda_2[i,j] != 0:
            Jcpl_1[i,j] = W1*np.cos(2*alpha)/(lambda_2[i,j]*(1 + lambda_1[i,j]**2/lambda_2[i,j]**2)) \
                        + W2*np.cos(4*alpha)/(lambda_1[i,j]*(1 + lambda_2[i,j]**2/lambda_1[i,j]**2)) \
                        - W1*np.sinc(2*alpha/np.pi)/(lambda_2[i,j]*(1 + lambda_1[i,j]**2/lambda_2[i,j]**2)) \
                        - W2*np.sinc(4*alpha/np.pi)/(lambda_1[i,j]*(1 + lambda_2[i,j]**2/lambda_1[i,j]**2))
            Jcpl_2[i,j] = W1*np.cos(2*alpha)/(lambda_1[i,j]*(1 + lambda_2[i,j]**2/lambda_1[i,j]**2)) \
                        - W2*np.cos(4*alpha)/(lambda_2[i,j]*(1 + lambda_1[i,j]**2/lambda_2[i,j]**2)) \
                        - W1*np.sinc(2*alpha/np.pi)/(lambda_1[i,j]*(1 + lambda_2[i,j]**2/lambda_1[i,j]**2)) \
                        + W2*np.sinc(4*alpha/np.pi)/(lambda_2[i,j]*(1 + lambda_1[i,j]**2/lambda_2[i,j]**2))
            Jde_1[i,j]  = lambda_2[i,j]*Jdia_1[i,j]/(2*lambda_1[i,j]) - lambda_1[i,j]*Jdia_2[i,j]/(2*lambda_2[i,j])
            Jde_2[i,j]  = lambda_2[i,j]*Jdia_1[i,j]/(2*lambda_1[i,j]) - lambda_1[i,j]*Jdia_2[i,j]/(2*lambda_2[i,j])
          Jcpl_3[i,j] = Jcpl_previous_3[i,j]*(np.cos(2*lambda_3[i,j]) - np.sinc(2*lambda_3[i,j]/np.pi))
          Jex_1[i,j]  = (Jdia_1[i,j] - Jdia_2[i,j])/2. + Jex_previous_1[i,j]
          Jex_2[i,j]  = (Jdia_1[i,j] - Jdia_2[i,j])/2. + Jex_previous_2[i,j]
          Jde_1[i,j]  += Jde_previous_1[i,j]
          Jde_2[i,j]  += Jde_previous_2[i,j]
          Jdia_0[i,j] += Jdia_previous_0[i,j]
          Jdia_1[i,j] += Jdia_previous_1[i,j]
          Jdia_2[i,j] += Jdia_previous_2[i,j]
          Jdia_3[i,j] += Jdia_previous_3[i,j]

    return Jdia_0,Jdia_1,Jdia_2,Jdia_3,Jcpl_0,Jcpl_1,Jcpl_2,Jcpl_3,Jde_1,Jde_2,Jex_1,Jex_2
