import numpy as np
from consts_rpmd import *
from potential import sys_Potential
from matrix_rpmd_vib import Matrix_RPMD_vib


def Evaluation_electronic_population(q,p, electronic_state):
    '''
    Pjj(t) = 1/electronic_N \sum_{1}^{electronic_N} ( 1/2 * (p^2 + q^2 -1 ) )
    :param q: shape[n_beads, n_electronic_state]. electronic mapping coordinate
    :param p: shape[n_beads, n_electronic_state]. electronic mapping momentum
    :param electronic_state: electronic state index j
    :return:
    '''

    q_j = q[:,electronic_state]
    p_j = p[:,electronic_state]

    P = 1/n_beads * np.sum( 1/2 * ( np.power(q_j , 2) + np.power(p_j , 2) - 1 ) )
    P = np.real(P)

    return P

def Evolve_interaction_part(q_,p_,Q_,P_ , V_sys):
    # evolve interaction part (exclude free ring polymer Hamiltonian)
    for bead_index in range(n_beads):
        # size [n_electronic_state]
        q_rate = V_sys.set_electronic_coordinate_rate(q_[bead_index], p_[bead_index], Q_[bead_index], P_[bead_index])
        p_rate = V_sys.set_electronic_momentum_rate(q_[bead_index], p_[bead_index], Q_[bead_index], P_[bead_index])
        P_rate = V_sys.set_nuclear_interaction_part_Force(q_[bead_index], p_[bead_index], Q_[bead_index],
                                                          P_[bead_index])

        q_[bead_index] = q_[bead_index] + dt / 2 * q_rate
        p_[bead_index] = p_[bead_index] + dt / 2 * p_rate
        P_[bead_index] = P_[bead_index] + dt / 2 * P_rate

    return q_, p_ , Q_, P_


def Evolve_system_evaluate_P(q_ , p_ , Q_ , P_ , electronic_state):
    '''
    Evolve system dynamics and compute probability at given electronic state
    :param q_ , p_ :  electronic initial coordinate and momentum : shape[n_beads, n_electronic_state]
    :param Q_ , P_ : nuclear coordinate and momentum : shape [n_beads]
    :param electronic_state : state to evaluate probability
    :return: Pjj: size: [nsteps_dynamics / nsteps_print]
    '''
    V_sys = sys_Potential()
    M_rpmd = Matrix_RPMD_vib()
    C_matrix = M_rpmd.trans_matrix()

    # normal mode [p,q]
    phase_point = np.zeros( (n_beads , 2) )

    Pjj_list = []

    for istep in range(nsteps_dynamics):
        # -------- evolution for interaction part ---------------
        q_, p_ , Q_, P_  = Evolve_interaction_part(q_, p_, Q_, P_, V_sys)

        # --------- evolve free ring polymer part -----------
        # transform to normal mode
        phase_point[:,0] = np.dot(P_ , C_matrix)
        phase_point[:,1] = np.dot(Q_,  C_matrix)
        # -------------- Evolution with normal mode --------------
        # omegak = 2 omega_N sin( bead_index * pi / n_beads)
        for bead_index in range(n_beads):
            phase_point[bead_index][:] = np.dot( M_rpmd.evol_matrix(omegak[bead_index]) , phase_point[bead_index][:] )

        # transform back to plain momentum and coordinate
        P_ = np.dot(C_matrix, phase_point[:,0])
        Q_ = np.dot(C_matrix, phase_point[:,1])

        # ------ evolution for interaction part --------
        q_, p_, Q_, P_ = Evolve_interaction_part(q_, p_, Q_, P_, V_sys)

        if(istep % nsteps_print == 0):
            Pjj = Evaluation_electronic_population(q_,p_, electronic_state)
            Pjj_list.append(Pjj)

    Pjj_list = np.array(Pjj_list)

    return Pjj_list
