'''
 @Author:	Adapted from code of Kai Song, ks838 _at_ cam.ac.uk

Chenghao Zhang : cz38 _at_ illinois.edu

 @Notes :   1. In this edition, we sample from the normal modes 
 			   space directly, without using the coordinate 
 			   transformation too many times (as presented in Ceriotti2010).

@Refs   :   1. Craig and David Manolopoulos, J. Chem. Phys. 121, 22  2004
		    2. Ceriotti et al. J. Chem. Phys. 133, 124104  2010 
'''
from __future__ import print_function

import os
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib

from initialize_distribution import initialize_dist
from Evolve_system import Evolve_system_evaluate_P
from consts_rpmd import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc= comm.Get_size()

print(__doc__)
	

# =========================================================================
#                  --------- EOM_Symplectic --------
# =========================================================================
def main():
	matplotlib.rcParams.update({'font.size': 20})
	file_path = "/home/phyzch/Presentation/4 point correlation/Tunneling prob/RPMD/"

	# electronic state to initialize
	electronic_state_init = 0
	# electronic state to compute P(t)
	electronic_state_eval = 0

	n_samplings_per_core = int(n_samplings / num_proc)
	print_num = int(nsteps_dynamics / nsteps_print)

	Pjj_list = []
	for i_sample in range(n_samplings_per_core):
		q_init, p_init, Q_init, P_init = initialize_dist(electronic_state_init)
		Pjj = Evolve_system_evaluate_P(q_init,p_init, Q_init, P_init, electronic_state_eval)
		Pjj_list.append(Pjj)

	# shape : [n_sampling_per_core , print_num]
	Pjj_list = np.array(Pjj_list)

	# broadcast and gather Pjj_list data
	data_type = type(Pjj_list[0][0])
	recv_Pjj = np.empty([num_proc, n_samplings_per_core, print_num] , dtype = data_type )

	comm.Gather(Pjj_list, recv_Pjj, 0)
	if(rank == 0):
		shape = recv_Pjj.shape
		Pjj_list = np.reshape(recv_Pjj , (shape[0] * shape[1] , shape[2]) )

		# average over sampling  shape : [ print_num ]
		Pjj_list_avg = np.mean(Pjj_list, 0)

		plot_Pjj_fig(Pjj_list_avg, file_path)
		save_data(Pjj_list_avg, file_path)

def plot_Pjj_fig(Pjj_list_avg , file_path):
	# time to print
	print_num = int(nsteps_dynamics / nsteps_print)
	t_print = np.array(range(print_num)) * dt

	# configure figure
	fig = plt.figure(figsize=(20, 10))
	spec = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
	spec.update(hspace=0.5, wspace=0.3)
	ax = fig.add_subplot(spec[0, 0])

	# plot result
	ax.plot(t_print, Pjj_list_avg, linewidth = 3)

	# set title and label
	ax.set_title("$P_{0}(t)$")
	ax.set_ylabel("$P_{0}$(t)")
	ax.set_xlabel('time')

	fig_name = "survival_prob.png"
	fig_name = os.path.join(file_path, fig_name)
	fig.savefig(fig_name)

def save_data(Pjj_list_avg , file_path):
	# time to print
	print_num = int(nsteps_dynamics / nsteps_print)
	t_print = np.array(range(print_num)) * dt

	file_name = "electron_state_prob.txt"
	file_name = os.path.join(file_path, file_name)

	with open(file_name , "w") as f:
		for t_index in range(print_num):
			f.write(str(t_print[t_index]) +" ")

		f.write("\n")

		for t_index in range(print_num):
			f.write(str(Pjj_list_avg[t_index]) + " ")

		f.write("\n")

