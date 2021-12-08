import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from consts_rpmd import beta
from potential import F,omega,Delta
from util import check_file_path_exist

def shifted_harmonic_oscillator_in_one_electronic_state_eig(harmonic_basis_N, params , spin_state , neigs ):
    N = harmonic_basis_N
    Delta, F, omega = params

    H = sparse.eye(N, N, format = 'lil')
    for i in range(harmonic_basis_N):
        H[i, i] = (i + 0.5) * omega

    # q * F * sigma_z
    if(spin_state == 0):
        # spin up.
        sign = 1
    elif (spin_state == 1):
        # spin down
        sign = -1
    else:
        print("wrong spin state")
        exit(-1)
        return

    for i in range(N -1 ):
        H[ i, i+1 ] =  F * np.sqrt(1 / (2 * omega)) * np.sqrt(i + 1) * sign
        H[i+1 , i ] = H [i, i+1 ]

    H = H.tocsr()

    [evl, evt] = sla.eigs(H, k=neigs, which='SR')

    sort_index = np.argsort(evl)
    evl = np.array(evl)[sort_index]
    evt = np.transpose(evt)
    evt = np.array([evt[sort_index[i]] for i in range(len(sort_index))])
    # evt may not be orthogonal to each other if eigenvalues are degenerate, try to solve this problem with Schmidth orthogonalization.
    for i in range(neigs - 1):
        evl1 = evl[i]
        evl2 = evl[i + 1]
        if (np.abs(evl1 - evl2) / np.abs(evl1) < 1e-3):
            # we need to use Schmidt orthogonalize vector
            evt3 = np.dot(evt[i], evt[i + 1]) / np.power(np.linalg.norm(evt[i]), 2)
            evt3 = evt3 * evt[i]
            evt[i + 1] = evt[i + 1] - evt3

            # check orthogonality
            if (np.abs(np.dot(evt[i + 1], evt[i])) > 1e-3):
                print('orthogonality not satisfied')

    for i in range(neigs):
        # normalize the eigen vectors
        Normalization = np.sqrt(np.sum(np.conj(evt[i, :]) * evt[i, :]))
        evt[i, :] = evt[i, :] / Normalization

        Normalization = np.sqrt(np.sum(np.conj(evt[i, :]) * evt[i, :]))
        if (np.abs(Normalization - 1) > 0.001):
            print('normalization failed')
        # eigen values MUST be real:
        evl = np.real(evl)

    return evl, evt

def solve_eigenstate_shifted_harmonic_oscillator(harmonic_basis_N, spin_state, param, eigenstate_num):
    '''
    solve the eigenstate for shifted harmonic oscillator for specific electronic state and find its representation on basis set.

    Actually this is a stupid code. We can actually solve harmonic oscillator at center then move it left/right hand side to get wave function representation in coordinateb basis set.
    :param harmonic_basis_N:  number of set for harmonic oscillator
    :param spin_state: spin state: 0 : spin up.   1 : spin down
    :param param:      Delta, F, omega = params
    :param eigenstate_num:  number of eigenstates to solve.
    :return:
    '''
    evl , evt = shifted_harmonic_oscillator_in_one_electronic_state_eig(harmonic_basis_N , param, spin_state, eigenstate_num)
    eigenstate_list = []
    for i in range(eigenstate_num):
        wave_func = np.zeros(2 * harmonic_basis_N)
        if(spin_state == 0):
            wave_func[: harmonic_basis_N] = np.real(evt[i] )
        else:
            wave_func[harmonic_basis_N: ] = np.real(evt[i])

        eigenstate_list.append(wave_func)

    eigenstate_list = np.array(eigenstate_list)
    return evl, eigenstate_list

def Schrodinger1D_spin_boson_harmonic_basis(harmonic_basis_N, params, neigs=20):
    '''
    evl, evt = Schrodinger1D_spin_boson_harmonic_basis(harmonic_basis_N=harmonic_basis_N, params=param, neigs=neigs)

    :param harmonic_basis_N: basis set number in each electronic state
    :param params: (Delta, F, omega)
    :param neigs: num of eigenstate to solve
    :return:
    '''
    N = 2 * harmonic_basis_N
    Delta, F, omega = params

    # set ground state energy = F^2 / 2m \omega^2 , such that bottom of well have V(x) = 0
    ground_state_energy = 0

    H = sparse.eye(N, N, format='lil')
    for i in range(harmonic_basis_N):
        H[i, i] = ground_state_energy + (i + 0.5) * omega
    for i in range(harmonic_basis_N):
        H[i + harmonic_basis_N, i + harmonic_basis_N] = ground_state_energy + (i + 0.5) * omega

    #  q * F * sigma_z
    for i in range(harmonic_basis_N - 1):
        H[i, i + 1] = F * np.sqrt(1 / (2 * omega)) * np.sqrt(i + 1)
        H[i + 1, i] = H[i, i + 1]

    for i in range(harmonic_basis_N - 1):
        H[i + harmonic_basis_N, i + harmonic_basis_N + 1] = - F * np.sqrt(1 / (2 * omega)) * np.sqrt(i + 1)
        H[i + harmonic_basis_N + 1, i + harmonic_basis_N] = H[i + harmonic_basis_N, i + harmonic_basis_N + 1]

    for i in range(harmonic_basis_N):
        H[i, i + harmonic_basis_N] = Delta
        H[i + harmonic_basis_N, i] = Delta

    # obtain neigs solutions from the sparse matrix
    [evl, evt] = sla.eigs(H, k=neigs, which='SR')

    sort_index = np.argsort(evl)
    evl = np.array(evl)[sort_index]
    evt = np.transpose(evt)
    evt = np.array([evt[sort_index[i]] for i in range(len(sort_index))])
    # evt may not be orthogonal to each other if eigenvalues are degenerate, try to solve this problem with Schmidth orthogonalization.
    for i in range(neigs - 1):
        evl1 = evl[i]
        evl2 = evl[i + 1]
        if (np.abs(evl1 - evl2) / np.abs(evl1) < 1e-3):
            # we need to use Schmidt orthogonalize vector
            evt3 = np.dot(evt[i], evt[i + 1]) / np.power(np.linalg.norm(evt[i]), 2)
            evt3 = evt3 * evt[i]
            evt[i + 1] = evt[i + 1] - evt3

            # check orthogonality
            if (np.abs(np.dot(evt[i + 1], evt[i])) > 1e-3):
                print('orthogonality not satisfied')

    for i in range(neigs):
        # normalize the eigen vectors
        Normalization = np.sqrt(np.sum(np.conj(evt[i, :]) * evt[i, :]))
        evt[i, :] = evt[i, :] / Normalization

        Normalization = np.sqrt(np.sum(np.conj(evt[i, :]) * evt[i, :]))
        if (np.abs(Normalization - 1) > 0.001):
            print('normalization failed')

    # eigen values MUST be real:
    evl = np.real(evl)

    return evl, evt

def nuclear_thermal_density_matrix( evt, evl_nuclear, evt_nuclear ):
    neigs = len(evt)

    nuclear_rho = np.zeros( (neigs, neigs) )

    partition_func = np.sum(np.exp(- beta * evl_nuclear ))
    for n in range(neigs):
        for m in range(neigs):
             nuclear_rho[n][m] =  np.sum (  np.exp(- beta * evl_nuclear ) / partition_func * np.matmul(evt_nuclear, evt[n]) * np.matmul( evt_nuclear, evt[m])   )

    return nuclear_rho

def evt_overlap_projection_operator( evt, harmonic_basis_N, spin_state_proj ):
    neigs = len(evt)
    evt_proj_overlap = np.zeros( (neigs, neigs) )
    for n in range(neigs):
        for m in range(neigs):
            if(spin_state_proj == 0):
                # spin state 0
                overlap = np.dot( evt[n][:harmonic_basis_N] , evt[m][:harmonic_basis_N] )
            else:
                # spin state 1
                overlap = np.dot( evt[n][harmonic_basis_N : ] , evt[m][harmonic_basis_N : ] )

            evt_proj_overlap[n][m] = overlap

    return evt_proj_overlap

def save_quantum_data(file_path, time, survival_prob):
    file_name = "quantum_survival_prob.txt"
    file_name = os.path.join(file_path, file_name )
    
    time_num = len(time)
    with open(file_name, "w") as f:
        for i in range(time_num):
            f.write( str(time[i]) + " ")
        f.write("\n")
        for i in range(time_num):
            f.write( str(survival_prob[i]) +" ")
        f.write("\n")

def plot_save_fig(file_path, time, survival_prob):
    # configure figure
    fig = plt.figure(figsize=(20, 10))
    spec = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
    spec.update(hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(spec[0, 0])

    # plot result
    ax.plot(time , survival_prob, linewidth=3)

    # set title and label
    ax.set_title("$P_{0}(t)$")
    ax.set_ylabel("$P_{0}$(t)")
    ax.set_xlabel('time')

    fig_name = "quantum_survival_prob.png"
    fig_name = os.path.join(file_path, fig_name)
    fig.savefig(fig_name)

def save_param(file_path):
    param_file = "quantum_param.txt"
    param_file = os.path.join(file_path, param_file)

    with open(param_file , "w") as f:
        f.write("beta: : " + str(beta) + "\n")
        f.write("[F , omega , Delta ] : " + str(F) + " , " + str(omega) + " , " + str(Delta) + "\n")

def solve_quantum_survival_probability():
    # solve survival probability of density matrix by resolving eigenstate spectrum of quantum system

    file_path = "/home/phyzch/Presentation/4 point correlation/Tunneling prob/RPMD/F=0.1 omega=1 Delta=1/beta=5/"

    check_file_path_exist(file_path)

    save_param(file_path)

    # parameter
    param = [Delta, F, omega]

    # basis set number
    harmonic_basis_N = 200

    # neigs : number of eigenstate to solve
    neigs_nuclear = int( 2 * 1/beta * 1/omega * 2 )
    neigs_nuclear = np.max( [neigs_nuclear, 50] )

    neigs = 2 * neigs_nuclear

    # spin_state
    init_spin_state = 0
    eval_spin_state = 0

    evl, evt = Schrodinger1D_spin_boson_harmonic_basis(harmonic_basis_N = harmonic_basis_N, params = param, neigs = neigs  )

    evl_nuclear, evt_nuclear = solve_eigenstate_shifted_harmonic_oscillator(harmonic_basis_N , init_spin_state, param, eigenstate_num = neigs_nuclear )

    # nuclear_rho  : nuclear density matrix on basis of evt
    nuclear_rho  = nuclear_thermal_density_matrix( evt, evl_nuclear, evt_nuclear )

    # <m|  |j><j| \otimes I |n>   Here |j><j| is electronic projection operator
    evt_proj_overlap = evt_overlap_projection_operator(evt, harmonic_basis_N, eval_spin_state)

    final_time = 10
    dt = 0.1
    time_num = int(final_time / dt)
    time = np.linspace(0,final_time, time_num )

    survival_prob = np.zeros((time_num) )
    
    for i in range(time_num):
        t = time[i]
        
        # \sum <n|nuclear_rho|m> * <m| evt_proj |n> e^{-i Em t/ \hbar}
        A = np.array ( [ np.sum( nuclear_rho[n] * evt_proj_overlap[ : ,n] * np.exp(-1j * evl * t) ) for n in range(neigs) ] )
        # \sum e^{i En t/ \hbar} * A[n]
        prob = np.sum( np.exp(1j * evl * t ) * A )
        
        survival_prob[i] = np.real(prob)

    # save data
    save_quantum_data(file_path,time, survival_prob)

    # save figure
    plot_save_fig(file_path, time, survival_prob)