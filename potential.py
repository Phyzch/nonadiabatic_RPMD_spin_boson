'''
 @Author:	Kai Song, ks838 _at_ cam.ac.uk

'''
from consts_rpmd import *

# F is displacement of harmonic oscillator : Fx
F = 6
# frequency of nuclear motion
omega = 1
# coupling between electronic state.
Delta = 1

def delta_nm(n, m):
	# simple delta function
	if(n == m ):
		delta = 1
	else:
		delta = 0
	return delta

class nuclear_Potential:
	# nuclear potential. Do not take free ring polymer potential $ M/(2 * beta_N^2 hbar^2) (R_alpha - R_(alpha + 1 ) )^2 $ into account

	# displacement of oscillatory motion
	R0 = F/( mass * pow(omega,2) )

	def set_pot(self,Q , n, m):
		V = 0
		if(n==0 and m==0):
			# Hamiltonian for electronic state 0
			V = 1/2 * mass * np.power(omega,2) * np.power(Q , 2) + F * Q
		elif (n==1 and m==1):
			# Hamiltonian for electronic state 1
			V = 1/2 * mass * np.power(omega,2) * np.power(Q  , 2) - F * Q
		elif( (n==0 and m==1) or (n==1 and m==0) ):
			V = Delta

		return V	
	# We always try to get the analytic derivative of potential V(q) 
	def set_force(self,Q , n, m):
		if( n==0 and m==0 ):
			Force = - mass * np.power(omega,2)* Q - F
		elif (n==1 and m==1 ):
			Force = - mass * np.power(omega, 2) * Q + F
		else:
			Force = 0

		return Force

class electronic_state_related_Potential:
	# potential related to electronic state. not include V0 (constant potential part)
	def set_pot(self,Q,n,m):
		V = 0
		if(n==0 and m==0):
			V = F * Q
		elif(n==1 and m==1):
			V = - F * Q
		elif( (n==0 and m==1) or (n==1 and m==0) ):
			V = Delta

		return V

	def set_force(self, Q, n, m ):
		# -\partial_{Q} Vnm(Q)
		if( n == 0 and m == 0 ):
			Force = -F
		elif( n == 1 and m == 1):
			Force = +F
		else:
			Force = 0

		return Force

	def set_V0(self, Q):
		# potential V0 common to all electronic state
		V0 = mass / 2 * np.power(omega,2) * np.power(Q,2)
		return V0

	def set_F0(self,Q):
		# F0 = - \partial_{R} V0
		F0 = - mass * np.power(omega,2) * Q
		return F0

class sys_Potential:
	# potential of nuclear + electronic state

	def set_electronic_momentum_rate(self , q, p, Q,P  , V_nuclear):
		'''
		see eq.18 in Chowdhury & Huo 2019
		:param q , p: electronic coordinate and momentum. shape : [ n_electronic_state ]
		:param Q: vib coordinate, shape:[dof]
		:param P: momentum, shape: [dof]
		q,p,Q,R should be the same bead.
		:return: dp/dt for electronic mapping momentum. size [n_electronic_state]
		'''

		momentum_rate = np.zeros(n_electronic_state)
		for n in range(n_electronic_state):
			p_rate = 0
			for m in range(n_electronic_state):
				p_rate = p_rate - V_nuclear.set_pot(Q,n,m) * q[m]

			momentum_rate[n] = p_rate

		return momentum_rate

	def set_electronic_coordinate_rate(self, q, p, Q, P , V_nuclear):
		'''
		see eq.18 in Chowdhury & Huo 2019
		:param q , p: electronic coordinate and momentum. shape : [ n_electronic_state ]
		:param Q: vib coordinate, shape:[dof]
		:param P: momentum, shape: [dof]
		q,p,Q,R should be the same bead.
		:return: dq/dt for electronic mapping momentum. size [n_electronic_state]
		'''

		coordinate_rate = np.zeros(n_electronic_state)
		for n in range(n_electronic_state):
			q_rate = 0
			for m in range(n_electronic_state):
				q_rate = q_rate + V_nuclear.set_pot(Q,n,m) * p[m]

			coordinate_rate[n] = q_rate

		return coordinate_rate

	def set_nuclear_interaction_part_Force(self , q, p, Q, P , V_nuclear ):
		'''
		See eq.(17) in Chowdhury & Huo 2019
		Here, we do not take evolution of free ring polymer part into consideration. we split Liouville operator to e^{-\Delta t L_V/2} e^{- \Delta t * L_0} e^{-\Delta t L_V/2}
		Thus we only take second and third term in eq.(17) into account
		:param q , p: electronic coordinate and momentum. shape : [ n_electronic_state ]
		:param Q: vib coordinate, shape:[dof]
		:param P: momentum, shape: [dof]
		q,p,Q,R should be the same bead.
		:return:
		'''

		Force = 0
		Force = Force + V_nuclear.set_F0(Q)
		for n in range(n_electronic_state):
			for m in range(n_electronic_state):
				Force = Force + 1/2 * V_nuclear.set_force(Q,n,m) * (q[n]*q[m] + p[n]*p[m] - delta_nm(n,m) )

		return Force



