import numpy as np
from nuclear_thermostat import initialize_thermal_nuclear_state
from consts_rpmd import *
from matrix_rpmd_vib import *
from potential import *

def initialize_electronic_state_dist(init_electron_index ):
    '''

    :param init_electron_index: initial state |i><i|.  init_electronic_index = i.
    :return: q_, p_ : mapping coordinate and momentum.  shape : [ n_beads, n_electronic_state]
    '''
    # MMST mapping coordinate q_, p_. See eq.(17) in Chowdhury and Huo 2019
    q_ = np.zeros(( n_beads , n_electronic_state))
    p_ = np.zeros(( n_beads ,  n_electronic_state))
    if(init_electron_index >= n_electronic_state ):
        raise("init electronic index error")

    for bead_index in range(n_beads):
        for elec_state_index in range(n_electronic_state):
            # action = p^2 + q^2
            if(elec_state_index == init_electron_index):
                action = np.sqrt(3)
            else:
                action = 1

            angle = np.random.random() * (2 * np.pi )

            q_[bead_index , elec_state_index] = action * np.cos(angle)
            p_[bead_index, elec_state_index ] = action * np.sin(angle)

    return q_ , p_

def initialize_dist(  init_electronic_index ):
    '''

    :param init_electronic_index: |i><i|
    :return: q_ , p_ : electronic initial coordinate and momentum : shape[n_beads, n_electronic_state]
    Q_ , P_ : nuclear coordinate and momentum : shape [n_beads]
    '''
    V_pot = nuclear_Potential()
    M_rpmd = Matrix_RPMD_vib()

    C_matrix = M_rpmd.trans_matrix()
    C_1, C_2 = M_rpmd.C_1_2()

    # nuclear coordinate and momentum
    # shape : [n_beads]
    Q_, P_ = initialize_thermal_nuclear_state(C_matrix, C_1, C_2, V_pot, M_rpmd, init_electronic_index )

    # electronic coordinate and momentum (MMST mapping)
    # shape : [n_beads, n_electronic_state]
    q_, p_ = initialize_electronic_state_dist(init_electronic_index)

    return q_ , p_, Q_, P_
