import numpy as np
from consts_rpmd import *
from init_condations import *
from potential import *
from matrix_rpmd_vib import *

def initialize_thermal_nuclear_state(C_matrix, C_1, C_2 , V_pot, M_rpmd , init_electron_index ):
    '''

    :param C_matrix: transformation matrix to normal mode coordinate for free ring polymer
    :param C_1 , C_2 : For thermostating. C1 = e^{- \Delta t / 2 * gamma }, C1^2 + C2^2  = 1.
    :param V_pot: potential class.
    :param M_rpmd: class. used to transform matrix and evolve nuclear motion
    :param init_electron_index: electronic state we choose to initialize our nuclear coordinate.
    :return: q_, p_ : coordinate and momentum of beads.
    '''
    phase_point = np.zeros( (n_beads, 2) )

    #  We first do thermalization to sample initial state ergodically. Then we simulate correlation function <AB(t)> without thermalstat (as with thermalstat, no real time info could be extracted.).
    # ****************************************************************
    #         		   	Step 1:  thermalization
    # ****************************************************************
    # 1.1 initialize forces, from q_, by V_pot.V_pot.set_force(q_)
    P_ = np.zeros(n_beads)
    Q_ = np.zeros(n_beads)
    for i_bead in range(n_beads):
        P_[i_bead] = p_gaussian()

    for istep in range(nsteps_equil):
        # transform to nornal modes
        phase_point[:, 0] = np.dot(P_, C_matrix)
        # ----------Ceriotti: Langevin step-----------
        for i_bead in range(n_beads):
            phase_point[i_bead, 0] = C_1[i_bead] * phase_point[i_bead, 0] + \
                                     np.sqrt(mass / beta_N) * C_2[i_bead] * xi_gaussian()
        # transform back to plain modes
        P_ = np.dot(C_matrix, phase_point[:, 0])
        P_ +=  equil_dt  / 2.0 * V_pot.set_force(Q_ , init_electron_index, init_electron_index )
        # transform to normal modes
        phase_point[:, 0] = np.dot(P_, C_matrix)
        phase_point[:, 1] = np.dot(Q_, C_matrix)
        # -------------Evolve normal modes------------
        # eq.(23) in 2010 paper.
        for i_bead in range(n_beads):
            phase_point[i_bead][:] = np.dot(M_rpmd.evol_matrix(omegak[i_bead] , equil_dt),
                                            phase_point[i_bead][:])
        # tansform back to plain momenta and coordinates
        P_ = np.dot(C_matrix, phase_point[:, 0])
        Q_ = np.dot(C_matrix, phase_point[:, 1])
        P_ += equil_dt  / 2.0 * V_pot.set_force(Q_ , init_electron_index, init_electron_index )
        # ----------Ceriotti: Langevin step-----------
        # transorm to normal modes, for PILET
        phase_point[:, 0] = np.dot(P_, C_matrix)
        for i_bead in range(n_beads):
            phase_point[i_bead, 0] = C_1[i_bead] * phase_point[i_bead, 0] + \
                                     np.sqrt(mass / beta_N) * C_2[i_bead] * xi_gaussian()
        P_ = np.dot(C_matrix, phase_point[:, 0])

        #		# -------------Andersen: collision------------
        #		for i_bead in range(n_beads):
        #			if np.random.uniform(0,1) <= dt* nu_poisson:
        #				p_[i_bead] = p_gaussian()
        # --------------END OF THERMALIZATION---------------

    return Q_, P_