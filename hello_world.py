from mpi4py import MPI
import numpy as np
from scipy.stats import ortho_group
import sklearn



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

trans_1 = None
trans_data_0 = None

data_dim = 100
local_dim = 50
data = np.random.uniform(0,1,size=(1000, 50))
received_data = np.zeros((1000, data_dim))
local_transformation = np.zeros((local_dim, data_dim))

# node id 0
if rank == 0:
    full_trans = ortho_group.rvs(100)

    trans_1 = full_trans[:50] 
    local_transformation = full_trans[50:]   
    transformed_data = data@local_transformation

    comm.Send(trans_1, dest=1, tag=11)

elif rank == 1:
    comm.Recv(local_transformation, source=0, tag=11)
    transformed_data = data@local_transformation


if rank == 0:
    comm.Send(transformed_data, dest=1, tag=21)

if rank == 1:
    comm.Recv(received_data, source = 0, tag=21)
    full_trans_data = received_data + transformed_data



