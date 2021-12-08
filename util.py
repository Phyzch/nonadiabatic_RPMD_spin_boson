import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc= comm.Get_size()

def check_file_path_exist(file_path):
    is_folder = os.path.isdir(file_path)
    if(not is_folder):
        raise("Folder does not exist.")


    # if(rank == 0):
    #     print("file path : " + str(file_path))
    #     val = input("please confirm this is file path you will store simulation result : Y/N \n")
    #     if (val == 'Y'):
    #         pass
    #     else:
    #         print("file path is not confirmed ")
    #
    #         comm.Abort()