'''
 @Author:	Kai Song, ks838 _at_ cam.ac.uk
 @Notes :   This part gives the constants and parameters.
'''
import numpy as np 


# parameters for the system
#the inverse temperature 
beta = 5 # a.u.

mass = 1.0

# ------ params for propagation ------
dt = 2 * pow(10,-3) # time step

equil_dt = pow(10,-2) # time step for thermalization of initial state
equil_time = 10
# steps for the equilibrating part
nsteps_equil = int(equil_time / equil_dt )

# steps for the dynamics
dynamics_time = 10
nsteps_dynamics = int(dynamics_time / dt )
print_time = 0.1
nsteps_print = int(print_time / dt )

# --------- for electronic state --------
n_electronic_state = 2

# -------- for the n beads ----------
# for simple potential forms (e.g., a double-well form), n_beads <10 are 
# engough. And, for harmonic form, n_beads = 1 is engough
n_beads = 16 # should be an even number in our settings
omega_N = n_beads/beta # we have used hbar = 1 . omega_N = 1/(beta_N * hbar)
beta_N  = beta/n_beads


# Normal mode frequency for free ring polymer.  See eq.(36) in Ceriotti et al. J. Chem. Phys. 133, 124104  2010.
omegak = np.zeros(n_beads)
for i_bead in range(n_beads):
	omegak[i_bead] = 2* omega_N * np.sin(i_bead*np.pi/n_beads)


#------ parameter for Ceriotti thermostatting-----
tau0 = 0.7 # an input parameter for tuning the efficiency

# The number of samplings (from the thermostatting).
# Typically, we need ~10^4 to get converged results.
# We started using a small number for testing
n_samplings = 100

# sampling initial momentum and coordinate.
mu_p = 0.0
sigma_p = np.sqrt(mass*n_beads/beta)
mu_q = 0.0



